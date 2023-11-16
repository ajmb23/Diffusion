from diffusion.architectures import create_model, NCSNpp, DDPM, MLP, bb_MLP
from diffusion.training.dataset import mult_datasets
from diffusion import VE_zero, VE, VP, sub_VP, setup_logger, \
                      load_checkpoint, load_checkpoint_ddp, \
                      save_checkpoint, save_checkpoint_ddp

import torch 
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam 
from datetime import timedelta
import logging
import os


def mult_gpu_setup():
    assert torch.distributed.is_available()

    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank 
    world_size = int(os.environ.get("SLURM_JOB_NUM_NODES")) * ngpus_per_node

    timeout = timedelta(seconds=180)

    torch.distributed.init_process_group(
        init_method = f'tcp://{os.environ.get("MASTER_ADDR")}:3456',
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timeout
    )
    return local_rank, rank, world_size


def mod_ema_opt_setup( device, arch_name, arch_params, ema_rate, lr, weight_decay=0, beta1=0.9, eps=1e-8, local_rank=None):
    #Initialize architecture, ema, optimizer
    call_model = create_model( arch_name)
    init_model = call_model( **arch_params )
    init_model = init_model.to( device )

    if local_rank is not None:
        init_model = DDP( init_model, device_ids=[local_rank] ) 

    init_ema = ExponentialMovingAverage(init_model.parameters(), decay=ema_rate)
    init_optimizer = Adam( init_model.parameters(), lr=lr, weight_decay=weight_decay,
                           betas=(beta1, 0.999), eps=eps )
    
    return init_model, init_ema, init_optimizer

def dataset_setup( config ):
    #Load data
    if config['data']['num_cond']>0:

        load_cond = []
        for i in range(1, config['data']['num_cond']+1 ):
            load_cond.append( config['data'][f'cond_data_{i}'] )

        data_sets = mult_datasets( config['data']['data_dir'], 
                                   config['data']['input_data'], *load_cond ) 

    else:
        data_sets = mult_datasets( config['data']['data_dir'], config['data']['input_data'] ) 

    return data_sets


def training_setup( config, local_rank=None, rank=None, world_size=None): 
    
    #Creates a file which gives infromation about the progress of training 
    #prog_file = open( 'training.txt', 'a' )
    #setup_logger(prog_file)

    logging.basicConfig( filename='training.txt', filemode='a', 
                         format='%(levelname)s - %(asctime)s - %(message)s', 
                         datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO )

    if local_rank is None:
        device = config['device']
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

    else:
        is_master = rank == 0
        device = torch.device("cuda", local_rank)
        logging.info(f"World size: {world_size}, global rank: {rank}, local rank: {local_rank}")

    #Initialize architecture, ema, optimizer
    init_model, init_ema, init_optimizer = mod_ema_opt_setup( device=device, arch_name=config['model']['name'], 
                                                              arch_params=config['model']['params'], 
                                                              ema_rate=config['model']['ema_rate'], 
                                                              lr=config['optimizer']['lr'], 
                                                              weight_decay=config['optimizer']['weight_decay'], 
                                                              beta1=config['optimizer']['beta1'], 
                                                              eps=config['optimizer']['eps'], 
                                                              local_rank=local_rank )

    #Dictionary saving initialized state of model, optim, ema and epochs
    init_state = dict( model=init_model, ema=init_ema, optimizer=init_optimizer, epoch=0 )

    #Load the SDE and initialize it with its min and max noise 
    sde = globals()[config['SDE']['name']]( config['SDE']['noise_min'], config['SDE']['noise_max'] )
    pert_mshift = sde.pert_mshift()
    pert_std = sde.pert_std()    

    #Get data
    data_sets = dataset_setup( config )
    
    #Create checkpoint directory and load checkpoint if it exists
    checkpoint_dir = os.path.join( config['training']['work_dir'], "checkpoints/")
    min_t = config['SDE']['min_t']
    max_t = config['SDE']['max_t']


    if local_rank is None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = load_checkpoint( checkpoint_dir, 'checkpoint.pth', init_state, device )

        logging.info( f"SDE:{config['SDE']['name']}, noise_min:{config['SDE']['noise_min']}, "
                    f"noise_max:{config['SDE']['noise_max']}, "
                    f"min_t:{min_t:.0e}, max_t:{max_t}" )

        dataloader = DataLoader( data_sets, batch_size=config['training']['batch_size'], 
                                 shuffle=config['training']['shuffle'], drop_last=False )
        
    else:
        if is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logging.info( f"SDE:{config['SDE']['name']}, noise_min:{config['SDE']['noise_min']}, "
                    f"noise_max:{config['SDE']['noise_max']}, "
                    f"min_t:{min_t:.0e}, max_t:{max_t}" )
            
        state = load_checkpoint_ddp( checkpoint_dir, 'checkpoint.pth', init_state, device )
        dataloader = DataLoader( data_sets, batch_size=config['training']['batch_size'], 
                                 shuffle=False, drop_last=False, 
                                 sampler=DistributedSampler(dataset=data_sets, 
                                                            shuffle=config['training']['shuffle']) )


    return device, state, checkpoint_dir, dataloader, \
           pert_mshift, pert_std, min_t, max_t 


def save_track_progress( config, state, epoch, sum_loss_iter, counter, checkpoint_dir, is_master=None ):

    state['epoch'] += 1

    if is_master is None:
        save_checkpoint( checkpoint_dir, 'checkpoint.pth', state )
        if epoch % config['training']['save_ckpt_rate'] == 0 or epoch == 0:
            avg_loss = sum_loss_iter/counter
            logging.info(f"epoch: {epoch}, training loss: {avg_loss.item():.2f}")
            save_checkpoint( checkpoint_dir, f'checkpoint_{epoch}.pth', state )

    else:
        if is_master: 
            save_checkpoint_ddp( checkpoint_dir, 'checkpoint.pth', state )

            if epoch % config['training']['save_ckpt_rate'] == 0 or epoch == 0:
                avg_loss = sum_loss_iter/counter
                logging.info(f"epoch: {epoch}, training loss: {avg_loss.item():.2f}")
                save_checkpoint_ddp( checkpoint_dir, f'checkpoint_{epoch}.pth', state )