import re
import os 
import json
import torch 
import logging
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import torch.nn.functional
import pickle

def load_config( config_file ):
    with open(config_file, 'r') as config_file:
        # Parse the JSON data
        config = json.load(config_file)

    return config

def save_checkpoint(dir_path, filename, state, local_rank=None):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'ema': state['ema'].state_dict(),
        'epoch': state['epoch']
        }
    
    if local_rank is None:
        saved_state['model'] = state['model'].state_dict()
    else:
        saved_state['model'] = state['model'].module.state_dict()

    checkpoint_file = os.path.join( dir_path, filename )
    torch.save(saved_state, checkpoint_file)

def find_highest_numbered_file(directory, file_extension):
    numbered_files = [
        file 
        for file in os.listdir(directory) 
        if file.endswith(file_extension) and any(char.isdigit() for char in file)
    ]

    if not numbered_files:
        return None

    highest_num = max(int(re.findall(r'\d+', file)[0]) for file in numbered_files)
    highest_file = next((file for file in numbered_files if str(highest_num) in file), None)
    return os.path.join(directory, highest_file)

def load_checkpoint(dir_path, state, device, local_rank=None):

    # Check for existing checkpoint, if there is one it loads it
    checkpoint_file = find_highest_numbered_file(dir_path, '.pth')

    if os.path.isfile( checkpoint_file ):

        loaded_state = torch.load(checkpoint_file, map_location=device)
        if local_rank is None:
            state['model'].load_state_dict(loaded_state['model'], strict=False)
        else:
            state['model'].module.load_state_dict(loaded_state['model'], strict=False)

        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['ema'].load_state_dict(loaded_state['ema'])
        state['epoch'] = loaded_state['epoch']
       
        logging.warning(f"Loaded checkpoint at epoch {state['epoch']} from {dir_path}.")
        state['epoch'] += 1
        
        return state
    else:
        logging.warning(f"No checkpoint found at {dir_path}. Starting from scratch.")
        return state

def load_arch_ema(dir_path, init_arch, init_ema, device):
    #Loads only the architecture
    checkpoint_file = find_highest_numbered_file(dir_path, '.pth')

    if os.path.isfile( checkpoint_file ):

        loaded_state = torch.load(checkpoint_file, map_location=device) 
        init_arch.load_state_dict(loaded_state['model'], strict=False)
        init_ema.load_state_dict(loaded_state['ema'])

        return init_arch, init_ema 
        
    else:
        ValueError( f"No checkpoint found in directory {dir_path}." )

def restart_checkpoint(dir_path, filename, state, device, local_rank=None):
    last_checkpoint_file = os.path.join( dir_path, filename )
    if os.path.isfile( last_checkpoint_file ):
        
        last_epoch = torch.load(last_checkpoint_file, map_location='cpu')['epoch']
        checkpoints = []
        pattern = re.compile(r"checkpoint_(\d+)\.pth")

        # Listing all files in the directory
        files = os.listdir(dir_path)

        # Extracting numbers from filenames using regex and storing them in the checkpoints list
        for file in files:
            match = pattern.match(file)
            if match:
                try:
                    checkpoint_number = int(match.group(1))
                    checkpoints.append(checkpoint_number)
                except ValueError:
                    pass

        # Sorting the checkpoints in descending order
        sorted_checkpoints = sorted(checkpoints, reverse=True)

        #If numbered checkpoint list has 2 or more elements
        if len(sorted_checkpoints) >= 2:
            if last_epoch > sorted_checkpoints[0]:
                logging.info(f"Last checkpoint epoch is {last_epoch}, " 
                             f"starting from numbered checkpoint at epoch {sorted_checkpoints[0]}.")
                return load_checkpoint(dir_path, f'checkpoint_{sorted_checkpoints[0]}.pth', state, device, local_rank=None)

            else:
                logging.info(f"Last checkpoint epoch is {last_epoch}, and last numbered checkpoint epoch is also {sorted_checkpoints[0]}. "
                             f"Starting from before last numbered checkpoint at epoch {sorted_checkpoints[1]}.")
                return load_checkpoint(dir_path, f'checkpoint_{sorted_checkpoints[1]}.pth', state, device, local_rank=None)
                       
        #If numbered checkpoint list has 1 element
        elif len(sorted_checkpoints) == 1:
            if last_epoch > sorted_checkpoints[0]:
                logging.info(f"Last checkpoint epoch is {last_epoch}, starting from numbered checkpoint at epoch {sorted_checkpoints[0]}.")
                return load_checkpoint(dir_path, f'checkpoint_{sorted_checkpoints[0]}.pth', state, device, local_rank=None)

            else:
                logging.info(f"Last checkpoint epoch is {last_epoch}, and last numbered checkpoint epoch is also {sorted_checkpoints[0]}. "
                             f"There are no previous numbered checkpoint, starting from scratch.")
                return state

        #If numbered checkpoint list is empty
        else:
            logging.info(f"No numbered checkpoint found. Starting from scratch.")
            return state

    else:
        logging.info(f"No checkpoint found in directory {dir_path}. Starting from scratch.")
        return state

def sigma_max(dataset):
    #compute larges eucledian distance in dataset
    pairwise_distances = cdist(dataset, dataset, metric='euclidean')
    highest_distance = np.max(pairwise_distances)
    return highest_distance

def sigma_max_torch( data_dic_file, device ):
    with open(data_dic_file, 'rb') as file:
        data_dic = pickle.load(file)
    
    flatten_data = []
    for key in data_dic:
        flatten_data.append( data_dic[key][1].flatten() )

    flatten_data = np.array(flatten_data)
    tensor_data = torch.tensor( flatten_data, device=device, dtype=torch.float32 )

    distance = torch.nn.functional.pdist(tensor_data, p=2)
    max_distance = torch.max( distance )
    return  max_distance.cpu().item()

def VE_samp_prob( sigma_min, sigma_max, num_steps, dim ):
    gamma = ( sigma_max / sigma_min )**(1/num_steps)
    c = (2 * dim)**(1/2)
    prob = norm.cdf(c * (gamma - 1) + 3 * gamma) - norm.cdf(c * (gamma - 1) - 3 * gamma)
    return prob