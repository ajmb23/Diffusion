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

def sample( config_file, idx_min, idx_max, sidx_min, sidx_max, cond_dic_file ):
    gpu_id = int(os.environ.get("SLURM_LOCALID"))
    time.sleep( 5*gpu_id )
    with open(cond_dic_file, 'rb') as file:
        cond_dic = pickle.load(file)

    config = load_config( config_file )
    device = config['device']

    batch_sizes = split_batch( sidx_min=sidx_min, sidx_max=sidx_max, 
                               batch_size=config['sampling']['batch_size'], 
                               ngpus=torch.cuda.device_count() ) 
    
    #global_sidxs = global_sidx( batch_sizes )[ int(os.environ.get("SLURM_LOCALID")) ]
    local_batch_sizes = batch_sizes[gpu_id]
    
    #Check if dictionnary exists or not
    os.makedirs(config['sampling']['sample_dir'], exist_ok=True)
    dic_name = f"{idx_min}_{idx_max}_{sum(local_batch_sizes)}_{gpu_id}.pkl"
    sample_dic_file = os.path.join( config['sampling']['sample_dir'], dic_name )

    if os.path.isfile( sample_dic_file ) is False:
        sample_dic = {}
    else:
        with open(sample_dic_file, 'rb') as file:
            sample_dic = pickle.load(file)

    #sample
    for sim_idx in range(idx_min, idx_max+1):
        for batch_size in local_batch_sizes:

            if sim_idx not in sample_dic:
                #If key doesn't exist create it
                cosmo = cond_dic[sim_idx][0]
                sample_dic[sim_idx] = [cosmo, None] 

            if sample_dic[sim_idx][1] is None or sample_dic[sim_idx][1].shape[0]<sum(local_batch_sizes): 
                cond_data = cond_dic[sim_idx][1]
                samples = sampling_batch( device, config, batch_size, cond_data )
                
                #Concatenate new samples with ones already in dictionary
                if sample_dic[sim_idx][1] is None:
                    sample_dic[sim_idx][1] = samples
                
                else:
                    sample_dic[sim_idx][1] = np.append( sample_dic[sim_idx][1], samples, axis=0 )
                
                with open(sample_dic_file, 'wb') as file:
                    pickle.dump(sample_dic, file)      