"""
Code from https://github.com/hassony2/torch_videovision
"""

import numbers

import random
import PIL.Image
import numpy as np
import PIL

from skimage.transform import resize, rotate
from skimage.util import pad
import torchvision.transforms.functional as TF

import warnings

from skimage import img_as_ubyte, img_as_float


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
            ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped

def crop_frame(clip, min_h, min_w, h, w):
    if isinstance(clip, np.ndarray):
        cropped = clip[min_h:min_h + h, min_w:min_w + w, :]

    elif isinstance(clip, PIL.Image.Image):
        cropped = clip.crop((min_w, min_h, min_w + w, min_h + h))
            
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip)))
    return cropped

def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')

def pad_frame(frame, h, w, im_h, im_w):
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(frame, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant', anti_aliasing=True) for img in clip
            ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=True, image=True):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip
        self.image = image

    def __call__(self, clip):
        if random.random() < 0.5 and self.horizontal_flip:
            # print("Horizontal flip Image")
            return np.fliplr(clip)
        
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)

        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, frame):
        """
        Args:
        img (PIL.Image or numpy.ndarray): Frame to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped frame
        """
        h, w = self.size
        if isinstance(frame, np.ndarray):
            im_h, im_w, im_c = frame.shape
        elif isinstance(frame, PIL.Image.Image):
            im_w, im_h = frame.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(frame)))

        frame = pad_frame(frame, h, w, im_h, im_w)
        im_h, im_w = frame.size[1:]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_frame(frame, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip, np.ndarray):
            rotated = rotate(image=clip, angle=angle, preserve_range=False)
        elif isinstance(clip, PIL.Image.Image):
            rotated = clip.rotate(angle)
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip)))

        return rotated


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip, np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: TF.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: TF.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: TF.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: TF.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms =  img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = clip
                for func in img_transforms:
                    jittered_img = func(jittered_img)#.astype('float32')
        
        elif isinstance(clip, PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: TF.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: TF.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: TF.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: TF.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to frame
            for func in img_transforms:
                jittered_img = func(clip)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip)))
        return jittered_img


class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
