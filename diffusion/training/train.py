from diffusion import load_config
from diffusion.training.setup import training_setup, save_track_progress, mult_gpu_setup
from diffusion.training.one_epoch import one_epoch
from diffusion import check_runtime

from torch.distributed import ReduceOp
import torch
import logging

def single( config_file ):
    config = load_config( config_file )

    device, state, checkpoint_dir, dataloader, \
    pert_mshift, pert_std, min_t, max_t = training_setup( config )

    for epoch in range ( state['epoch'] , config['training']['num_epochs'] ):

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
        
        if torch.isnan(sum_loss_iter):
            logging.info("Model exploded and returns NaN stopping training")
            break

        check_runtime( 30, save_track_progress, config, state, epoch, 
                       sum_loss_iter, counter, checkpoint_dir )


def mult( config_file ):
    config = load_config( config_file )
    
    local_rank, rank, world_size = mult_gpu_setup()

    device, state, checkpoint_dir, dataloader, \
    pert_mshift, pert_std, min_t, max_t = training_setup( config, local_rank=local_rank, 
                                                          rank=rank, world_size=world_size )

    for epoch in range ( state['epoch'] , config['training']['num_epochs'] ):

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
        
        check_runtime( 30, save_track_progress, config, state, epoch, 
                       sum_loss_iter, counter, checkpoint_dir, rank==0 )