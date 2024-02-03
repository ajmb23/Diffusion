from diffusion.architectures import create_model, NCSNpp, DDPM, MLP, bb_MLP
from diffusion.sampling.sampling_fn import samplers
from diffusion import VE_zero, VE, VP, sub_VP, load_arch_ema, load_config
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm 
import numpy as np 
import pickle
import torch
import os 

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

def sampling_batch( config, device ):
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

    init_sampler = samplers( score_model=score_model, ema=ema, 
                            batch_size=config['sampling']['batch_size'], 
                            dim=config['sampling']['dim'], 
                            pred_num_steps=config['sampling']['pred_steps'], 
                            mean=config['sampling']['mean'], 
                            std=config['sampling']['std'], 
                            first_t=config['sampling']['start_t'], 
                            last_t=config['sampling']['end_t'], 
                            pert_std=pert_std, 
                            drift_coeff=drift_coeff, 
                            diffusion_coeff=diffusion_coeff, 
                            device=device )

    samples = init_sampler.sample( config['sampling']['tqdm_bool'] )
    np_samples = samples.detach().cpu().numpy()
    return np_samples

def split_idx(idx_min, idx_max, ngpu):
    # Calculate total range and interval size
    total_range = idx_max - idx_min + 1
    base_interval = total_range // ngpu
    remainder = total_range % ngpu

    # Initialize lists to hold the split values
    split_lists = [[] for _ in range(ngpu)]

    # Distribute values into lists
    current_idx = idx_min
    for i in range(ngpu):
        # Calculate interval size for this list
        interval = base_interval + (1 if i < remainder else 0)

        # Add values to the current list
        split_lists[i] = list(range(current_idx, current_idx + interval))

        # Update current index
        current_idx += interval

    return split_lists

def sample( config_file, idx_min, idx_max ):
    config = load_config( config_file )
    device = config['device']

    sim_idx_list = split_idx( idx_min=idx_min, idx_max=idx_max, 
                             ngpu=torch.cuda.device_count() ) [ int(os.environ.get("SLURM_LOCALID")) ]
    
    #Check if dictionnary exists or not
    os.makedirs(config['sampling']['sample_dir'], exist_ok=True)
    dic_name = f"{min(sim_idx_list)}_{max(sim_idx_list)}_{config['sampling']['dict_name']}"
    sample_dic_file = os.path.join( config['sampling']['sample_dir'], dic_name )

    if os.path.isfile( sample_dic_file ) is False:
        sample_dic = {}
    else:
        with open(sample_dic_file, 'rb') as file:
            sample_dic = pickle.load(file)

    #key of dictionnary is sim number, sample for each sim
    for sim_idx in sim_idx_list:
        print(sim_idx)

        #If sim number is not yet a key create it
        if sim_idx not in sample_dic:
            sample_dic[sim_idx] = []

        #Check number of arrays in list at index, if less than total_samp/batch_size then add more samples
        i = len( sample_dic[sim_idx] ) 
        while i < config['sampling']['total_samp'] /  config['sampling']['batch_size']:

            samples = sampling_batch( config, device )
            sample_dic[sim_idx].append(samples)

            with open(sample_dic_file, 'wb') as file:
                pickle.dump(sample_dic, file)
           
            i+=1