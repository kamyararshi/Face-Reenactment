import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

from torchvision import transforms
from PIL import Image
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class ImagesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, use_augmentation=False, dataset_amount=None, device='cpu'):
        """
        Inputs:
            root: path to the folder of dataset (train/test)
        """
        self.root = root_dir
        self.is_train = is_train
        self.data_dir = os.path.join(self.root, "train" if self.is_train else "test") # depending on whether we want the train or test set
        self.img_size = frame_shape
        self.total_len = 0
        self.augmentation_params = augmentation_params
        self.use_augmentation = use_augmentation
        self.id_sampling = id_sampling
        self.DEVICE = device
    
        self.classes_list = os.listdir(self.data_dir)
        self.num_classes = len(self.classes_list)
        if dataset_amount != None:
            self.classes_list = os.listdir(self.data_dir)[:dataset_amount]
            self.num_classes = len(self.classes_list)
        
        # Pre process transfomrs
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            #Resize(size=(self.img_size[0], self.img_size[1])), #IDK Why it warns to put str
        ])

        # Calculate the total number of frames available in root
        for _, _, files in os.walk(self.root, topdown=False):
            self.total_len += len(files)

        # Augmentations #TODO: Check
        if self.is_train and (self.augmentation_params!=None) and self.use_augmentation:
            print("Augmentation Active")
            self.transform = AllAugmentationTransform(**augmentation_params)

        
    def __len__(self):
        return self.num_classes

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            raise NotImplementedError("ID Sampling not implemented yet")
            # name = self.videos[idx]
            # path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            random_class = self.classes_list[idx]
            
        images_list = os.listdir(os.path.join(self.data_dir, random_class))
        
        idx_1, idx_2 = random.sample(range(0, len(images_list)), 2)
        
        sample1_path = os.path.join(self.data_dir, random_class, images_list[idx_1])
        sample2_path = os.path.join(self.data_dir, random_class, images_list[idx_2])
        
        # Reads the image and returns tensor in shape H, W, C
        img1_raw = Image.open(sample1_path)
        img2_raw = Image.open(sample2_path)


        out = {}
        if self.is_train and self.use_augmentation:
            out['driving']  = self.transform(self.pre_transform(img1_raw))
            out['source'] = self.transform(self.pre_transform(img2_raw))

        elif (self.is_train and not self.use_augmentation) or not self.is_train:
            out['driving']  = self.pre_transform(img1_raw)
            out['source'] = self.pre_transform(img2_raw)
        else:
            # video = np.array(video_array, dtype='float32')
            # out['video'] = video.transpose((3, 0, 1, 2))
            # out['video']=images_list
            raise ValueError("Invalid Mode. Should be either train or test")

        out['name'] = random_class

        return out

class AllFrames(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root = root_dir
        self.all = []
        for i, _, k in os.walk(self.root, topdown=False):
            self.all += [os.path.join(i, name) for name in k]

        # Pre process transfomrs
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            #Resize(size=(self.img_size[0], self.img_size[1])), #IDK Why it warns to put str
        ])
        
        self.len = len(self.all)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = Image.open(self.all[idx])
        return self.pre_transform(img)

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
