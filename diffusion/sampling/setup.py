from diffusion.architectures import create_model, NCSNpp, DDPM, MLP, bb_MLP
from diffusion.sampling.sampling_fn import samplers
from diffusion import VE_zero, VE, VP, sub_VP, load_arch_ema, load_config
from torch_ema import ExponentialMovingAverage
from itertools import accumulate
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

def sampling_batch( device, config, batch_size ):
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
                            batch_size=batch_size, 
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

def split_batch(sidx_min, sidx_max, batch_size, ngpus):
    #Based on sidx_min, sidx_max, batch_size and ngpus it splits up the different
    #sizes over the different number of gpus to get the total amount samples
    #returns list with ngpus of sublist containing that gpus batch sizes
    total_samples = sidx_max-sidx_min+1
    base_samples_per_gpu, remainder = divmod(total_samples, ngpus)
    samples_per_gpu = [base_samples_per_gpu + 1 if i < remainder else base_samples_per_gpu for i in range(ngpus)]
    
    batches_per_gpu = [[batch_size] * (sample // batch_size) + ([sample % batch_size] if sample % batch_size else [])
                      for sample in samples_per_gpu]
    
    return batches_per_gpu

def global_sidx( split_sdix_list ):
    #keeps track of the total samples that batch size on that gpu created
    flat_list = [element for sublist in zip(*split_sdix_list) for element in sublist]
    new_list = list(accumulate(flat_list))
    # Reconstructing the list in the specified format
    num_sublists = len(split_sdix_list)
    reconstructed_list = [[new_list[i] for i in range(j, len(new_list), num_sublists)] for j in range(num_sublists)]

    return reconstructed_list

def sample( config_file, idx_min, idx_max, sidx_min, sidx_max,  ):
    config = load_config( config_file )
    device = config['device']

    batch_sizes = split_batch( sidx_min=sidx_min, sidx_max=sidx_max, 
                               batch_size=config['sampling']['batch_size'], 
                               ngpus=torch.cuda.device_count() ) 
    
    global_sidxs = global_sidx( batch_sizes )[ int(os.environ.get("SLURM_LOCALID")) ]
    local_batch_sizes = batch_sizes[ int(os.environ.get("SLURM_LOCALID")) ]
    
    #Check if dictionnary exists or not
    os.makedirs(config['sampling']['sample_dir'], exist_ok=True)
    dic_name = f"{idx_min}_{idx_max}_{global_sidxs[0]-config['sampling']['batch_size']}_{global_sidxs[-1]}.pkl"
    sample_dic_file = os.path.join( config['sampling']['sample_dir'], dic_name )

    if os.path.isfile( sample_dic_file ) is False:
        sample_dic = {}
    else:
        with open(sample_dic_file, 'rb') as file:
            sample_dic = pickle.load(file)

    #sample
    for sim_idx in range(idx_min, idx_max+1):
        for i, batch_size in enumerate(local_batch_sizes):
            print(sim_idx)
            print(global_sidxs[i]-batch_size+1)
            print(global_sidxs[i])
            if (sim_idx, global_sidxs[i]-batch_size+1, global_sidxs[i]) not in sample_dic:
                samples = sampling_batch( device, config, batch_size )
                sample_dic[(sim_idx, global_sidxs[i]-batch_size+1, global_sidxs[i])] = samples

            with open(sample_dic_file, 'wb') as file:
                pickle.dump(sample_dic, file)      
