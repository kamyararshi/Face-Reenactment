import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
import imageio

import os
from typing import Dict
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt', writer=None, ddp=False):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.global_epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.writer = writer
        self.ddp = ddp


    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def log_tensorboard(self, state: str, losses: dict, generated: torch.tensor, gt: torch.tensor, global_epoch: int) -> None:
        for key, value in losses.items():
            self.writer.add_scalar(f'{state}/{key}', value, global_epoch)
        
        metrics = self.compute_metrics(generated, gt)
        # Log scalar metrics
        for metric, value in metrics.items():
            self.writer.add_scalar(f'{state}/Metric/{metric}', value, global_epoch)
        self.writer.flush()
        
    def compute_metrics(self, generated: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        metrics = {
            "SSIM": ssim(generated, gt).item(),
            "PSNR": psnr(generated, gt).item(),
        }
        return metrics
    
    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        # source, transformed, driving, pred, occlusion, mask0(no color), mask1, mask2, mask3, ..., mask15 
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        if self.ddp:
            cpk = {k: v.state_dict() for k, v in self.models.items()}
        else:
            cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk['global_epoch'] = self.global_epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def move_optimizer_states(optimizer, device):
        """
        Moves all state tensors in an optimizer to the specified device.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose state needs to be moved.
            device (str or torch.device): The device to move the state tensors to.
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, he_estimator=None,
                optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_he_estimator=None,
                rank=None):
        """
        Loads a checkpoint and maps it to the correct device for DDP.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            generator, discriminator, kp_detector, he_estimator: Model components to load.
            optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator: Optimizers to load.
            map_location (str): Device to map the checkpoint to. Default is None.
            rank (int): Rank of the current process. Used to map devices in DDP.

        Returns:
            tuple: epoch and global_epoch from the checkpoint.
        """
        if rank is not None:
            map_location = f"cuda:{rank}"  # Map checkpoint to the rank-specific GPU

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load models
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint['discriminator'])
            except KeyError:
                print('No discriminator in the state-dict. Discriminator will be randomly initialized.')

        # Load optimizers and move states
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            Logger.move_optimizer_states(optimizer_generator, map_location)

        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
                Logger.move_optimizer_states(optimizer_discriminator, map_location)
            except KeyError:
                print('No discriminator optimizer in the state-dict. Optimizer will not be initialized.')

        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
            Logger.move_optimizer_states(optimizer_kp_detector, map_location)

        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])
            Logger.move_optimizer_states(optimizer_he_estimator, map_location)

        # Return epoch and global epoch for resuming training
        return checkpoint['epoch'], checkpoint['global_epoch']


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        if self.writer is not None:
            self.writer.close()
        if self.ddp:
            destroy_process_group()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, global_epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        self.global_epoch = global_epoch
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        # self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'][:, :, :2].data.cpu().numpy()     # 3d -> 2d
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'][:, :, :2].data.cpu().numpy()   # 3d -> 2d
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'][:, :, :2].data.cpu().numpy()    # 3d -> 2d
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        ## Mask
        if 'mask' in out:
            for i in range(out['mask'].shape[1]):
                mask = out['mask'][:, i:(i+1)].data.cpu().sum(2).repeat(1, 3, 1, 1)    # (n, 3, h, w)
                # mask = F.softmax(mask.view(mask.shape[0], mask.shape[1], -1), dim=2).view(mask.shape)
                mask = F.interpolate(mask, size=source.shape[1:3]).numpy()
                mask = np.transpose(mask, [0, 2, 3, 1])

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['mask'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))
                
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
