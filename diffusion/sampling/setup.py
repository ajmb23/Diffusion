from diffusion.architectures import create_model, NCSNpp, DDPM, MLP, bb_MLP
from diffusion.sampling.sampling_fn import samplers
from diffusion import VE_zero, VE, VP, sub_VP, load_arch_ema, load_config
from torch_ema import ExponentialMovingAverage
from itertools import accumulate
from tqdm import tqdm 
import numpy as np 
import logging
import pickle
import torch
import os 
import time 

def mod_ema_setup( device, dir_path, filename, arch_name, arch_params, ema_rate):
    #Initialize architecture, ema
    call_model = create_model( arch_name)
    init_model = call_model( **arch_params )
    init_model = init_model.to( device )

    init_ema = ExponentialMovingAverage(init_model.parameters(), decay=ema_rate)

    score_model, ema = load_arch_ema( dir_path=dir_path, filename=filename, init_arch=init_model, 
                                      init_ema=init_ema, device=device )
    
    score_model.eval()

    return score_model, ema

def sampling_batch( device, config, batch_size, *cond_img ):
    #Takes care of loading checkpoint, config parameters, and doing the sampling
    #Returns tensors of samples set on batch size
    
    checkpoint_dir = os.path.join( config['training']['work_dir'], "checkpoints/")
    score_model, ema = mod_ema_setup(  device=device, dir_path=checkpoint_dir,
                                       filename=config['sampling']['ckpt_filename'], 
                                       arch_name=config['model']['name'],
                                       arch_params=config['model']['params'],
                                       ema_rate=config['model']['ema_rate'] )
    
 
    sde = globals()[config['SDE']['name']]( config['SDE']['noise_min'], config['SDE']['noise_max'] )
    drift_coeff = sde.drift_coeff()
    diffusion_coeff = sde.diffusion_coeff() 
    pert_std = sde.pert_std()  

    init_sampler = samplers(score_model, ema, batch_size, 
                            config['sampling']['dim'], 
                            config['sampling']['pred_steps'], 
                            config['sampling']['mean'], 
                            config['sampling']['std'], 
                            config['sampling']['start_t'], 
                            config['sampling']['end_t'], 
                            pert_std, drift_coeff, 
                            diffusion_coeff, device,
                            None, None, None, cond_img )

    samples = init_sampler.sample( config['sampling']['tqdm_bool'] )
    np_samples = samples.detach().cpu().numpy()
    return np_samples

def repeat_elements(list, N):
    #repeats elements of a list N times one after the other
    return [item for item in list for _ in range(N)]

def create_sublists(lst, N):
    """
    Seperates elements in N sublists based on position 0, N, 2N
    for first sublists, 0+1, N+1, 2N+1 for second sublist and so on
    """
    sublists = [[] for _ in range(N)]  # Create N empty sublists
    for i, num in enumerate(lst):
        index = i % N  # Determine which sublist to append to based on remainder
        sublists[index].append(num)
    return sublists

def idx_per_gpu( idx_min, idx_max, n_gpu_per_idx, n_gpu ):
    """
    Creates lists of simulations indexes per gpu based on 
    the number of gpus needed for the total amount of samples
    """
    idx_list = list(range(idx_min, idx_max + 1))
    repeat_idx = repeat_elements( idx_list, n_gpu_per_idx )
    idx_list_per_gpu = create_sublists( repeat_idx, n_gpu )
    return idx_list_per_gpu

def count_repeat(lst):
    """
    Looks at elements in list and determines number of 
    times each element gets repeated at each position 
    even if same value at different positions. List 
    returned is the same shape as input list.
    """
    repeated_elements = []
    for i in range( len(lst) ):
        element = lst[i]
        count = 0    
        for num in lst:
            if num == element:
                count += 1 
        repeated_elements.append(count)
    
    return repeated_elements

def sample( config_file, idx_min, idx_max, sidx_min, sidx_max, cond_dic_file ):
    #Load conditional data
    gpu_id = int(os.environ.get("SLURM_LOCALID"))
    time.sleep( 5*gpu_id )
    with open(cond_dic_file, 'rb') as file:
        cond_dic = pickle.load(file)

    #Load config file
    config = load_config( config_file )
    device = torch.device("cuda", gpu_id)
    batch_size = config['sampling']['batch_size']

    #List of idx for each gpu
    total_samples = int( sidx_max - sidx_min +1 )
    n_gpu_per_idx = int( total_samples/batch_size )
    n_gpu = int( torch.cuda.device_count() )
    idx_list = idx_per_gpu(idx_min, idx_max, n_gpu_per_idx, n_gpu )[gpu_id]
    
    #List of total samples per idx per gpu
    factor = count_repeat(idx_list)
    tot_samp_per_gpu = (total_samples/n_gpu_per_idx) * np.array(factor)

    #Check if sampling dictionnary exists or not
    os.makedirs( config['sampling']['sample_dir'], exist_ok=True )
    dic_name = f"{idx_min}_{idx_max}_{sidx_min}_{sidx_max}_{gpu_id}.pkl"
    sample_dic_file = os.path.join( config['sampling']['sample_dir'], dic_name )

    if os.path.isfile( sample_dic_file ) is False:
        sample_dic = {}
    else:
        with open(sample_dic_file, 'rb') as file:
            sample_dic = pickle.load(file)

    #Track progress with logging file 
    logging.basicConfig( filename=f'sample_{idx_min}_{idx_max}_{sidx_min}_{sidx_max}.txt', filemode='a', 
                         format='%(levelname)s - %(asctime)s - %(message)s', 
                         datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO )

    #sample
    for track, sim_idx in enumerate(idx_list):
        
        logging.info(f'GPU {gpu_id}: sim_idx {sim_idx}')
        
        if sim_idx not in sample_dic:
            #If key doesn't exist create it
            cosmo = cond_dic[sim_idx][0]
            sample_dic[sim_idx] = [cosmo, None] 

        if sample_dic[sim_idx][1] is None or sample_dic[sim_idx][1].shape[0]<tot_samp_per_gpu[track]: 
            cond_data = cond_dic[sim_idx][1]
            samples = sampling_batch( device, config, batch_size, cond_data )
            
            #Concatenate new samples with ones already in dictionary
            if sample_dic[sim_idx][1] is None:
                sample_dic[sim_idx][1] = samples
            
            else:
                sample_dic[sim_idx][1] = np.append( sample_dic[sim_idx][1], samples, axis=0 )
            
            #Save progress
            with open(sample_dic_file, 'wb') as file:
                pickle.dump(sample_dic, file)  

def merge_dictionaries(in_dir, out_dir):
    merged_dict = {}

    # Iterate over files in the directory
    for filename in os.listdir(in_dir):
        filepath = os.path.join(in_dir, filename)
        
        # Check if the file is a pickle
        if os.path.isfile(filepath) and filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Merge dictionaries
            for key in data.keys():
                if key in merged_dict:
                    merged_dict[key][1] = np.append( merged_dict[key][1], data[key][1], axis=0 )
    
                else:
                    merged_dict[key] = [ data[key][0], data[key][1] ]

    #Save merged dictionary
    out_filename = f'samples_{min(merged_dict.keys())}_{max(merged_dict.keys())}.pkl'
    out_file = os.path.join(out_dir, out_filename)
    with open(out_file, 'wb') as f:
        pickle.dump(merged_dict, f)