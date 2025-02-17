import torch
from torchvision import transforms
from torchvision import models
import modules.hopenet as hopenet
from modules.model import headpose_pred_to_degree

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import pickle
import csv

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
        self.class_hist_data = defaultdict(lambda: defaultdict(lambda: torch.zeros(num_bins)))

    def update_histogram(self, yaw, pitch, roll, classes):
        """
        Update histograms with new batch data.

        Args:
            yaw, pitch, roll: Tensors containing rotation values.
            classes: List of class names for each sample in the batch.
        """
        for angle, name in zip([yaw, pitch, roll], ["yaw", "pitch", "roll"]):
            # Update total histogram
            counts = torch.histc(angle, bins=self.num_bins, min=-180, max=180)
            self.hist_data[name] += counts.cpu()

            # Update per-class histograms
            for i, class_name in enumerate(classes):
                class_angle = angle[i]
                class_counts = torch.histc(class_angle, bins=self.num_bins, min=-180, max=180)
                self.class_hist_data[class_name][name] += class_counts.cpu()

    def update_csv(self, paths, classes, yaw, pitch, roll):
        """
        Update a CSV file with the paths and rotation values.

        Args:
            paths: List of paths to the frames.
            yaw, pitch, roll: Tensors containing rotation values.
        """
        with open("experiments/rotation_data.csv", "a", newline='') as f:
            writer = csv.writer(f)
            
            for path, cls, y, p, r in zip(paths, classes, yaw, pitch, roll):
                writer.writerow([path, cls, y.item(), p.item(), r.item()])

    @torch.no_grad()
    def process_batch(self, frames, classes, paths):
        """
        Process a batch of frames and update the histogram.

        Args:
            frames: Tensor of shape (batch_size, channels, height, width).
            classes: List of class names for each sample in the batch.
            paths: List of paths to the frames.
        """
        frames = frames.to(self.device)
        
        yaw, pitch, roll = self.hopenet(self.transform_hopenet(frames))
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        self.update_histogram(yaw, pitch, roll, classes)
        self.update_csv(paths, classes, yaw, pitch, roll)

    def process_all(self, dataloader):
        """
        Process all frames in a dataloader and update the histogram.

        Args:
            dataloader: Dataloader containing frames and class names.
        """
        for frame, classes, paths in tqdm(dataloader, desc="Processing frames for rotation histogram"):
            self.process_batch(frame, classes, paths)

    def plot_histogram(self, save_path: str):
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
        plt.savefig(save_path)

    def plot_histogram_per_class(self, save_path: str):
        """
        Plot the final histograms for yaw, pitch, and roll per class.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        angle_names = ["yaw", "pitch", "roll"]
        for ax, name in zip(axes, angle_names):
            bins = torch.linspace(-180, 180, steps=self.num_bins + 1)
            for class_name, class_hist in self.class_hist_data.items():
                ax.bar(bins[:-1].numpy(), class_hist[name].numpy(), width=360 / self.num_bins, align='edge', alpha=0.5, label=class_name)
                ax.set_title(f"{name.capitalize()} Histogram")
                ax.set_xlabel("Angle (degrees)")
                ax.set_ylabel("Frequency")
                ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)

    def save_histograms(self, file_path: str):
        """
        Save the total and per-class histograms to a file.
        """
        # Ensure conversion to standard dictionary
        data_to_save = {
            "total_histogram": {k: list(v) for k, v in self.hist_data.items()},  # Convert mutable data types
            "class_histograms": {cls: {k: list(v) for k, v in hist.items()} for cls, hist in self.class_hist_data.items()}
        }

        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load_histograms(cls, file_path: str):
        """
        Load the total and per-class histograms from a file.
        """
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
        
        instance = cls(hopenet_snapshot=None)  # Create an instance without initializing hopenet
        instance.hist_data = defaultdict(lambda: torch.zeros(instance.num_bins), loaded_data["total_histogram"])
        instance.class_hist_data = defaultdict(lambda: defaultdict(lambda: torch.zeros(instance.num_bins)), loaded_data["class_histograms"])
        
        return instance

if __name__ == "__main__":
    from experiments import RotationHistogram
    from frames_dataset import AllFrames
    from torch.utils.data import DataLoader
    import yaml
    import sys
    import os

    config_path = sys.argv[1]
    device = torch.device(sys.argv[2]) if len(sys.argv) == 1 else torch.device("cuda:0")
    print("Device used is:", device)
    os.makedirs("experiments/", exist_ok=True)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    experiment = RotationHistogram(config['train_params']['hopenet_snapshot'], device=device)
    all_frames = AllFrames(config['dataset_params']['root_dir'])
    all_loader = DataLoader(all_frames, batch_size=32, shuffle=False, num_workers=16)
    
    experiment.process_all(all_loader)
    experiment.plot_histogram(save_path="experiments/rotation_histogram.png")
    experiment.save_histograms(file_path="experiments/histograms.pkl")
    experiment.plot_histogram_per_class(save_path="experiments/class_rotation_histogram.png")

    # Example of loading histograms
    loaded_experiment = RotationHistogram.load_histograms(file_path="experiments/histograms.pkl")
    loaded_experiment.plot_histogram("experiments/loaded_rotation_histogram.png")
