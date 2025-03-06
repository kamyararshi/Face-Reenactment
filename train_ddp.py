import torch.optim
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import trange, tqdm
import os
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

def ddp_setup(rank: int, world_size: int, backend: str='nccl', MASTER_ADDR: str='localhost', MASTER_PORT: str='12366') -> None:
    import os
    os.environ['MASTER_ADDR'] = MASTER_ADDR # Change this to the IP of the master node
    os.environ['MASTER_PORT'] = MASTER_PORT # Random free port
    init_process_group(backend, rank=rank, world_size=world_size)
    
def train(rank: int, world_size: int, config: dict,
            generator: nn.Module, discriminator: nn.Module, kp_detector: nn.Module, he_estimator: nn.Module,
            opt: dict, log_dir: str, dataset: Dataset, val_dataset: Dataset, writer: bool=True) -> None:
    train_params = config['train_params']
    device = rank
    ddp_setup(rank=device, world_size=world_size)

    optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.AdamW(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.AdamW(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))

    # TODO: Weight loading for DDP is not Working yet
    if opt.checkpoint is not None:
        start_epoch, global_epoch = Logger.load_cpk(opt.checkpoint, generator, discriminator, kp_detector, he_estimator,
                                      optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator, rank=rank)
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
    writer = SummaryWriter(f'{log_dir}/runs/') if writer and rank==0 else None

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq'], writer=writer, ddp=True) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs'], initial=start_epoch, total=train_params['num_epochs']):
            for x in tqdm(dataloader):
                global_epoch += 1
                x = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in x.items()}
                generator_full.train()
                losses_generator, generated = generator_full(x, add_expression=opt.add_expr, rec_driving=opt.rec_driv) #TODO: add_expressions=False is used to avoid the expression loss

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
                if rank == 0: # Only the master node logs the losses' scores
                    logger.log_iter(losses=losses, epoch=epoch, global_epoch=global_epoch)
                    logger.log_scores(logger.names)
                    # Tensorboard TODO: Add more metrics like psnr, ssim, etc.
                    if writer is not None:
                        logger.log_tensorboard('train', losses, generated['prediction'][-1].detach(), x['driving'].detach(), global_epoch)

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
                _ = run_validation(generator_full, val_loader, device, epoch, logger, writer)

            

@torch.no_grad()
def run_validation(generator_full, val_loader, device, epoch, logger, writer=None) -> None:
    for vaL_batch in val_loader:
        vaL_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in vaL_batch.items()}
        generator_full.eval()
        losses_generator, generated = generator_full(vaL_batch, add_expression=False)

        losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
        # Tensorboard TODO: Add more metrics like psnr, ssim, etc.
        if writer is not None:
            logger.log_tensorboard('val', losses, generated['prediction'][-1].detach(), vaL_batch['driving'].detach(), epoch)

    cross_input, out = Logger.cross_reenactment(val_loader, generator_full, device)
    save_path = os.path.join(logger.visualizations_dir, "%s-crossreen.png" % str(epoch).zfill(logger.zfill_num))
    #src, drv, corssreenact
    Logger.plot_images(cross_input, out, save_path=save_path)
    return vaL_batch, generated