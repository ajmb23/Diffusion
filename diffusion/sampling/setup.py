from diffusion.architectures import create_model, NCSNpp, DDPM, MLP, bb_MLP
from diffusion.sampling.sampling_fn import samplers
from diffusion import VE_zero, VE, VP, sub_VP, load_arch_ema, load_config
from torch_ema import ExponentialMovingAverage
import numpy as np 
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


def sample( config_file ):
    config = load_config( config_file )
    device = config['device']

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
    np_samples = samples.cpu().detach().numpy()
    np.save( config['training']['work_dir']+'samples.npy', np_samples )