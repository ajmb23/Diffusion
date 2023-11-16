import torch 
import logging
import os 
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import signal

def load_config( config_file ):
    with open(config_file, 'r') as config_file:
        # Parse the JSON data
        config = json.load(config_file)

    return config

def save_checkpoint(dir_path, filename, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'epoch': state['epoch']
        }
    checkpoint_file = os.path.join( dir_path, filename )
    torch.save(saved_state, checkpoint_file)

def save_checkpoint_ddp (dir_path, filename, state):
    saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].module.state_dict(),
    'ema': state['ema'].state_dict(),
    'epoch': state['epoch']
    }
    checkpoint_file = os.path.join( dir_path, filename )
    torch.save(saved_state, checkpoint_file)

def load_checkpoint(dir_path, filename, state, device):

    # Check for existing checkpoint, if there is one it loads it
    checkpoint_file = os.path.join( dir_path, filename )

    if os.path.isfile( checkpoint_file ):

        loaded_state = torch.load(checkpoint_file, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['epoch'] = loaded_state['epoch']

        logging.warning(f"Loaded checkpoint at epoch {state['epoch']} from {dir_path}.")
        return state
    else:
        logging.warning(f"No checkpoint found at {dir_path}. Starting from scratch.")
        return state

def load_checkpoint_ddp(dir_path, filename, state, device):

    # Check for existing checkpoint, if there is one it loads it
    checkpoint_file = os.path.join( dir_path, filename )

    if os.path.isfile( checkpoint_file ):

        loaded_state = torch.load(checkpoint_file, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['epoch'] = loaded_state['epoch']

        logging.warning(f"Loaded checkpoint at epoch {state['epoch']} from {dir_path}.")
        return state
    else:
        logging.warning(f"No checkpoint found at {dir_path}. Starting from scratch.")
        return state


def load_arch_ema(dir_path, filename, init_arch, init_ema, device):
    #Loads only the architecture
    checkpoint_file = os.path.join( dir_path, filename )

    if os.path.isfile( checkpoint_file ):

        loaded_state = torch.load(checkpoint_file, map_location=device) 
        init_arch.load_state_dict(loaded_state['model'], strict=False)
        init_ema.load_state_dict(loaded_state['ema'])

        return init_arch, init_ema 
        
    else:
        ValueError( f"No checkpoint found in directory {dir_path} with name {filename}" )

def sigma_max(dataset):
    #compute larges eucledian distance in dataset
    pairwise_distances = cdist(dataset, dataset, metric='euclidean')
    highest_distance = np.max(pairwise_distances)
    return highest_distance

def VE_samp_prob( sigma_min, sigma_max, num_steps, dim ):
    gamma = ( sigma_max / sigma_min )**(1/num_steps)
    c = (2 * dim)**(1/2)
    prob = norm.cdf(c * (gamma - 1) + 3 * gamma) - norm.cdf(c * (gamma - 1) - 3 * gamma)
    return prob

def setup_logger(prog_file):
    handler = logging.StreamHandler(prog_file)
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s', datefmt=time_format)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def check_runtime(time_limit, function, *args):

    def handler(signum, frame):
        raise TimeoutError("Function Took Too Long")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)

    try:
        function(*args)
    finally:
        signal.alarm(0)  