import torch
from torchvision import transforms
from torchvision import models
import modules.hopenet as hopenet
from modules.model import headpose_pred_to_degree

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

class RotationHistogram:
    def __init__(self, hopenet_snapshot: str, num_bins: int=50, device: str="cuda"):
        """
        Class to compute and plot histograms of yaw, pitch, and roll incrementally.

        Args:
            hopenet: Pre-trained head-pose estimation model.
            num_bins: Number of bins for the histogram.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """
        self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        print('Loading hopenet')
        hopenet_state_dict = torch.load(hopenet_snapshot)
        self.hopenet.load_state_dict(hopenet_state_dict)
        if torch.cuda.is_available() and device!=None:
            self.hopenet = self.hopenet.to(device)
            self.hopenet.eval()
        self.transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.num_bins = num_bins
        self.device = device
        self.hist_data = defaultdict(lambda: torch.zeros(num_bins))

    def update_histogram(self, yaw, pitch, roll):
        """
        Update histograms with new batch data.

        Args:
            yaw, pitch, roll: Tensors containing rotation values.
        """
        for angle, name in zip([yaw, pitch, roll], ["yaw", "pitch", "roll"]):
            # Calculate bin indices for the current batch
            counts = torch.histc(angle, bins=self.num_bins, min=-180, max=180)
            self.hist_data[name] += counts.cpu()

    @torch.no_grad()
    def process_batch(self, frames):
        """
        Process a batch of frames and update the histogram.

        Args:
            frames: Tensor of shape (batch_size, channels, height, width).
        """
        frames = frames.to(self.device)
        
        yaw, pitch, roll = self.hopenet(self.transform_hopenet(frames))
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        self.update_histogram(yaw, pitch, roll)

    def process_all(self, dataloader):
        """
        Process all frames in a dataloader and update the histogram.

        Args:
            dataloader: Dataloader containing frames.
        """
        for batch in tqdm(dataloader, desc="Processing frames for rotation histogram"):
            self.process_batch(batch)

    def plot_histogram(self):
        """
        Plot the final histograms for yaw, pitch, and roll.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        angle_names = ["yaw", "pitch", "roll"]
        for ax, name in zip(axes, angle_names):
            bins = torch.linspace(-180, 180, steps=self.num_bins + 1)
            ax.bar(bins[:-1].numpy(), self.hist_data[name].numpy(), width=360 / self.num_bins, align='edge')
            ax.set_title(f"{name.capitalize()} Histogram")
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("rotation_histogram.png")


if __name__ == "__main__":
    from experiments import RotationHistogram
    from frames_dataset import AllFrames
    from torch.utils.data import DataLoader
    import yaml
    import sys

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    experiment = RotationHistogram(config['train_params']['hopenet_snapshot'])
    all_frames = AllFrames(config['dataset_params']['root_dir'])
    all_loader = DataLoader(all_frames, batch_size=32, shuffle=False, num_workers=16)
    
    experiment.process_all(all_loader)
    experiment.plot_histogram()