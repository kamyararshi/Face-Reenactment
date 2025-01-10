from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import trange, tqdm
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
# import torchmetrics #TODO: Install this and use it for metrics

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from frames_dataset import DatasetRepeater

def ddp_setup(rank: int, world_size: int, backend: str='nccl') -> None:
    import os
    os.environ['MASTER_ADDR'] = 'localhost' # Change this to the IP of the master node
    os.environ['MASTER_PORT'] = '12355' # Random free port
    init_process_group(backend, rank=rank, world_size=world_size)
    
def train(rank: int, world_size: int, config: dict,
            generator: nn.Module, discriminator: nn.Module, kp_detector: nn.Module, he_estimator: nn.Module,
            checkpoint: str, log_dir: str, dataset: Dataset, val_dataset: Dataset, writer: bool=True) -> None:
    train_params = config['train_params']
    map_location = f"cuda:{rank}"
    device = rank
    ddp_setup(rank=device, world_size=world_size)

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    # TODO: Weight loading for DDP is not Worling yet
    if checkpoint is not None:
        start_epoch, global_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, he_estimator,
                                      optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator, map_location)
    else:
        start_epoch, global_epoch = 0, 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=16, drop_last=False, sampler=DistributedSampler(dataset))
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=16, sampler=DistributedSampler(val_dataset))

    generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'], device=device).to(device)
    generator_full = DDP(generator_full, device_ids=[device], find_unused_parameters=True) # Wrap the model with DDP
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params, device=device).to(device)
    discriminator_full = DDP(discriminator_full, device_ids=[device], find_unused_parameters=True) # Wrap the model with DDP

    # Tensorboard
    writer = SummaryWriter(f'{log_dir}/runs/') if writer and get_rank()==0 else None

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq'], writer=writer, ddp=True) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in tqdm(dataloader):
                global_epoch += 1
                x = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in x.items()}
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_he_estimator.step()
                optimizer_he_estimator.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward(retain_graph=False)
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                logger.log_scores(logger.names)
                # Tensorboard TODO: Add more metrics like psnr, ssim, etc.
                if writer!=None:
                    for key, value in losses.items():
                        writer.add_scalar(f'train/{key}', value, global_epoch)
                        writer.flush()

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_he_estimator.step()
            
            if rank == 0: # Only the master node saves the checkpoint
                logger.log_epoch(epoch, global_epoch,
                                    {'generator': generator,
                                        'discriminator': discriminator,
                                        'kp_detector': kp_detector,
                                        'he_estimator': he_estimator,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,
                                        'optimizer_kp_detector': optimizer_kp_detector,
                                        'optimizer_he_estimator': optimizer_he_estimator}, inp=x, out=generated)
        
        # Validation
        run_validation(generator, val_loader, device, epoch, writer)

    destroy_process_group()
            

def run_validation(generator, val_loader, device, epoch, writer) -> None:
    for vaL_batch in val_loader:
        vaL_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in vaL_batch.items()}
        losses_generator, generated = generator(vaL_batch)

        losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
        # Tensorboard TODO: Add more metrics like psnr, ssim, etc.
        if writer!=None:
            for key, value in losses.items():
                writer.add_scalar(f'val/{key}', value, epoch)
                writer.flush()