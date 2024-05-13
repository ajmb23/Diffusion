from diffusion import load_config
from diffusion.training.setup import training_setup, save_track_progress, mult_gpu_setup
from diffusion.training.one_epoch import one_epoch

from torch.distributed import ReduceOp
import logging
import signal
import torch
import os 

def single( config_folder, config_file ):
    config_path = os.path.join(config_folder, config_file)
    config = load_config( config_path )

    device, state, checkpoint_dir, dataloader, \
    pert_mshift, pert_std, min_t, max_t = training_setup(config)

    for epoch in range ( state['epoch'] , config['training']['num_epochs']+1 ):

        sum_loss_iter, counter = one_epoch(
                                    device=device, 
                                    dataloader=dataloader, 
                                    score_model=state['model'], 
                                    optimizer=state['optimizer'], 
                                    ema=state['ema'], 
                                    pert_mshift=pert_mshift, 
                                    pert_std=pert_std, 
                                    min_t=min_t, 
                                    max_t=max_t, 
                                    grad_clip=config['optimizer']['grad_clip'], 
                                    data_noise=config['data']['data_noise'],
                                    cond_noise=config['data']['cond_noise'],
                                    bit=config['data']['bit']
                                    )
        
        if torch.isnan(sum_loss_iter):
            logging.info("Model exploded and returns NaN stopping training")
            break
        
        save_track_progress( config, state, epoch, sum_loss_iter, counter, checkpoint_dir )

def mult( config_folder, config_file ):
    config_path = os.path.join(config_folder, config_file)
    config = load_config( config_path )
    
    local_rank, rank, world_size = mult_gpu_setup()

    device, state, checkpoint_dir, dataloader, \
    pert_mshift, pert_std, min_t, max_t = training_setup( config, local_rank=local_rank, 
                                                          rank=rank, world_size=world_size )

    for epoch in range ( state['epoch'] , config['training']['num_epochs']+1 ):

        sum_loss_iter, counter = one_epoch(
                                    device=device, 
                                    dataloader=dataloader, 
                                    score_model=state['model'], 
                                    optimizer=state['optimizer'], 
                                    ema=state['ema'], 
                                    pert_mshift=pert_mshift, 
                                    pert_std=pert_std, 
                                    min_t=min_t, 
                                    max_t=max_t, 
                                    grad_clip=config['optimizer']['grad_clip'], 
                                    cond_noise=config['data']['cond_noise']
                                    )
        
        torch.distributed.reduce( sum_loss_iter, dst=0, op=ReduceOp.SUM)
        torch.distributed.reduce( counter, dst=0, op=ReduceOp.SUM)

        if torch.isnan(sum_loss_iter):
            logging.info("Model exploded and returns NaN stopping training")
            break
        
        save_track_progress( config, state, epoch, sum_loss_iter, counter, checkpoint_dir, local_rank, rank==0 )