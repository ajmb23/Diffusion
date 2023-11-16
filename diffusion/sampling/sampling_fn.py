import torch
from tqdm import tqdm 


class samplers():

  def __init__(self, score_model, ema, batch_size, dim, 
               pred_num_steps, mean, std, first_t, last_t, 
               pert_std, drift_coeff, diffusion_coeff, device):
    
    self.score_model = score_model
    self.ema = ema
    
    self.B = batch_size
    self.D = dim
    self.pred_num_steps = pred_num_steps
    self.mean = mean
    self.std = std
    self.first_t = first_t
    self.last_t = last_t
    
    self.pert_std = pert_std
    self.drift_coeff = drift_coeff
    self.diffusion_coeff = diffusion_coeff    
    self.device = device

  
  def setup(self):

    init_x = self.mean + self.std*torch.randn( [self.B]+self.D, device=self.device )
    time_steps = torch.linspace(self.first_t, self.last_t, self.pred_num_steps, device=self.device)
    dt = (self.first_t-self.last_t)/self.pred_num_steps

    return time_steps[1:], dt, init_x


  def EM_update(self, x, dt, time_step):

    f = x*self.drift_coeff(time_step)
    g = self.diffusion_coeff( torch.ones_like(x)*time_step )
    score = self.score_model( torch.ones([self.B], device=self.device)*time_step, x )/self.pert_std(time_step)
    
    if time_step > self.last_t:
        x_mean = x + ( g**2 * score - f ) * dt
        x_noisy = x_mean +  g * torch.randn_like(x) * dt**0.5
        return x_noisy
    
    if time_step==self.last_t:
        x_final =  x + ( g**2 * score - f ) * dt
        return x_final


  def sample(self, tqdm_bool):

    time_steps, dt, x = self.setup()

    if tqdm_bool == True:
        time_steps = tqdm( time_steps )

    with torch.no_grad(), self.ema.average_parameters():

        for time_step in time_steps:
            x = self.EM_update( x, dt, time_step )
        return x