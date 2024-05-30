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

def load_dic(filepath):
    with open(filepath, 'rb') as file:
        dic = pickle.load(file)
    return dic

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

    if checkpoint_file is not None:

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

    if checkpoint_file is not None:

        loaded_state = torch.load(checkpoint_file, map_location=device) 
        init_arch.load_state_dict(loaded_state['model'], strict=False)
        init_ema.load_state_dict(loaded_state['ema'])

        return init_arch, init_ema 
        
    else:
        ValueError( f"No checkpoint found in directory {dir_path}." )

def sigma_max(dataset):
    #compute larges eucledian distance in dataset
    pairwise_distances = cdist(dataset, dataset, metric='euclidean')
    highest_distance = np.max(pairwise_distances)
    return highest_distance

def sigma_max_torch( data_file, device, dic=True ):
    
    if dic:
        data_dic = load_dic( data_file )
        
        flatten_data = []
        for key in data_dic:
            flatten_data.append( data_dic[key][1].flatten() )

        flatten_data = np.array(flatten_data)
    
    else:
        data_array = np.load(data_file)
        flatten_data = data_array.reshape(data_array.shape[0], -1)

    tensor_data = torch.tensor( flatten_data, device=device, dtype=torch.float32 )

    distance = torch.nn.functional.pdist(tensor_data, p=2)
    max_distance = torch.max( distance )
    return  max_distance.cpu().item()

def VE_samp_prob( sigma_min, sigma_max, num_steps, dim ):
    gamma = ( sigma_max / sigma_min )**(1/num_steps)
    c = (2 * dim)**(1/2)
    prob = norm.cdf(c * (gamma - 1) + 3 * gamma) - norm.cdf(c * (gamma - 1) - 3 * gamma)
    return prob