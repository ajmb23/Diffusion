import torch
from tqdm import tqdm 


class samplers():

  def __init__(self, score_model, ema, batch_size, dim:list, 
               pred_num_steps, mean, std, first_t, last_t, 
               pert_std, drift_coeff, diffusion_coeff, device, 
               num_corr_steps=None, corr_step_type=None, 
               corr_step_size=None, *cond):
    
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

    self.num_corr_steps = num_corr_steps
    self.corr_step_type = corr_step_type
    self.corr_step_size = corr_step_size
    
    self.cond = []
    if cond is not None:
        self.cond = torch.tensor(*cond[0]).reshape(1, *dim).repeat(batch_size, *[1]*len(dim)).to(device)
  
  def setup(self):

    init_x = self.mean + self.std*torch.randn( [self.B]+self.D, device=self.device )
    time_steps = torch.linspace(self.first_t, self.last_t, self.pred_num_steps, device=self.device)
    dt = (self.last_t-self.first_t)/self.pred_num_steps

    return time_steps[1:], dt, init_x


  def EM_update(self, x, dt, time_step, *cond):
    
    f = x*self.drift_coeff(time_step)
    g = self.diffusion_coeff( torch.ones_like(x)*time_step )
    score = self.score_model( torch.ones([self.B], device=self.device)*time_step, x, *cond )/self.pert_std(time_step)
    
    if time_step > self.last_t:
        x_mean = x + ( f - g**2 * score ) * dt
        x_noisy = x_mean +  g * torch.randn_like(x) * (-dt)**0.5
        return x_noisy
    
    if time_step==self.last_t:
        x_final =  x + ( f - g**2 * score ) * dt
        return x_final
    
    
  def langevin_update(self, x, time_step, step_size, step_type=None):
    #This looks wrong why is time torch.ones ?
    score = self.score_model( torch.ones([self.B], device=self.device)*time_step, x )/self.pert_std(time_step)
    if step_type=="pert_std":
      step_size = step_size * self.pert_std( time_step )    

    for i in range(self.num_corr_steps):
      x = x + step_size*score + (2*step_size)**0.5 * torch.randn_like(x)     
    return x    


  def sample(self, tqdm_bool):

    time_steps, dt, x = self.setup()

    if tqdm_bool == True:
        time_steps = tqdm( time_steps )

    with torch.no_grad(), self.ema.average_parameters():
        
        if self.num_corr_steps is None or self.num_corr_steps==0: 
          for time_step in time_steps:
              x = self.EM_update( x, dt, time_step, self.cond )
          return x
        
        else:
          for time_step in time_steps:
            x = self.EM_update( x, dt, time_step )
            x = self.langevin_update( x, time_step, step_size=self.corr_step_size, step_type=self.corr_step_type )
          return x