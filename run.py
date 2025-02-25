import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset, ImagesDataset

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector, HEEstimator

import torch

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "generate"])
    parser.add_argument("--gen", default="original", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--no_expr", dest="add_expr", action="store_false", help="Include expression deformation in keypoints")
    parser.add_argument("--rec_driv", dest="rec_driv", action="store_true", help="Reconstruct the driving feature volume")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    parser.set_defaults(add_expr=True)
    parser.set_defaults(rec_driv=False)

    opt = parser.parse_args()
    if opt.checkpoint is not None:
        # find the yaml file at the same directory as the checkpoint
        yaml_file = [i for i in os.listdir(os.path.dirname(opt.checkpoint)) if i.endswith('.yaml')][0]
        config_path = os.path.join(os.path.dirname(opt.checkpoint), yaml_file)
        if os.path.exists(config_path):
            opt.config = config_path
            print(f"Using config file at {opt.config}")
        else:
            print(f"Using config file at {opt.config}")

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device_id = opt.device_ids[0]

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime()) + '_device_' + str(device_id)
        

    if opt.gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif opt.gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if torch.cuda.is_available():
        print('cuda is available')
        generator.to(device_id)
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(device_id)
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(device_id)

    if opt.verbose:
        print(kp_detector)

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])

    if torch.cuda.is_available():
        he_estimator.to(device_id)

    if opt.mode == 'train':
        # dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
        dataset = ImagesDataset(is_train=True, **config['dataset_params'])
        val_dataset = ImagesDataset(is_train=False, **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        if len(opt.device_ids)>1 and torch.cuda.device_count()>1:
            from train_ddp import train
            import torch.multiprocessing as mp
            print("Training with DDP...")
            world_size = len(opt.device_ids)
            mp.spawn(train, args=(world_size, config, generator, discriminator, kp_detector, he_estimator, opt, log_dir, dataset, val_dataset),
                     nprocs=world_size, join=True)
        else:
            from train import train
            print("Training with Single GPU...")
            train(config, generator, discriminator, kp_detector, he_estimator, opt, log_dir, dataset, val_dataset, device_id)
