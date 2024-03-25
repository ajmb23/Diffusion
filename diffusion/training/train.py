from diffusion import load_config
from diffusion.training.setup import training_setup, save_track_progress, mult_gpu_setup
from diffusion.training.one_epoch import one_epoch

from torch.distributed import ReduceOp
import logging
import signal
import torch
import os 

def runtime_restart( time_limit, config_folder, config_file, restart):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                if torch.cuda.device_count() > 1:
                    torch.distributed.destroy_process_group()
                    mult(config_folder, config_file, restart)
                else:
                    single(config_folder, config_file, restart)

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(time_limit)

            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return wrapper

    return decorator

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
                                    cond_noise=config['data']['cond_noise'],
                                    bit=config['training']['bit']
                                    )
        
        if torch.isnan(sum_loss_iter):
            logging.info("Model exploded and returns NaN stopping training")
            break
        
        save_track_progress( config, state, epoch, sum_loss_iter, counter, checkpoint_dir )

def mult( config_folder, config_file, restart=False ):
    config_path = os.path.join(config_folder, config_file)
    config = load_config( config_path )
    
    local_rank, rank, world_size = mult_gpu_setup()

    decorated_training_setup = runtime_restart(300, config_folder, config_file, restart=True)(training_setup)
    device, state, checkpoint_dir, dataloader, \
    pert_mshift, pert_std, min_t, max_t = decorated_training_setup( config,restart=restart,
                                                                    local_rank=local_rank, rank=rank, 
                                                                    world_size=world_size )

    for epoch in range ( state['epoch'] , config['training']['num_epochs']+1 ):

        decorated_one_epoch = runtime_restart(300, config_folder, config_file, restart=True)(one_epoch)
        sum_loss_iter, counter = decorated_one_epoch(
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
        
        decorated_save_track_progess = runtime_restart(60, config_folder, config_file, restart=True)(save_track_progress)
        decorated_save_track_progess( config, state, epoch, sum_loss_iter, counter, checkpoint_dir, local_rank, rank==0 )