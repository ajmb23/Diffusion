import torch 
import numpy as np
from torch_ema import ExponentialMovingAverage

def forward_sde( x0, drift, diffusion, t_start, t_end, num_steps, num_real, device ):
    dt = (t_end-t_start)/num_steps
    realizations = torch.zeros( [num_steps, num_real], device=device )

    t = torch.linspace(t_start, t_end, num_steps, device=device)
    t_all = t.unsqueeze(0).expand(num_real, -1)

    normal_samples = torch.randn_like(t_all)
    dW = normal_samples * ( dt )**0.5 

    x = x0*torch.ones_like(t_all).to(device)
    for j in range(1, num_steps):
        x[:,j] = x[:, j - 1] + x[:, j-1] * drift( t_all[:, j-1] ) * dt + diffusion( t_all[:, j-1] ) * ( dW[:, j-1] )     
        
    realizations = x.transpose(1,0)
    return t, realizations

def forward_sde_dist( x0, drift, diffusion, t_start, t_end, num_steps, device ):
    #Make sure inputs are column wise not row wise [samples, 1]
    
    dt = (t_end-t_start)/num_steps
    realizations = torch.zeros( [num_steps, len(x0)], device=device )

    t = torch.linspace(t_start, t_end, num_steps, device=device)
    t_all = t.unsqueeze(0).expand( len(x0) , -1)

    normal_samples = torch.randn_like(t_all)
    dW = normal_samples * ( dt )**0.5

    first_x = x0.to(device)
    other_x = torch.zeros([len(x0), num_steps-1], device=device)
    x = torch.cat([first_x, other_x], dim=1)
    for j in range(1, num_steps):
        x[:,j] = x[:, j - 1] + x[:, j-1] * drift( t_all[:, j-1] ) * dt + diffusion( t_all[:, j-1] ) * ( dW[:, j-1] )     
        
    realizations = x.transpose(1,0)
    return t, realizations

def reverse_sde( mean, std, drift, diffusion, score_fn, t_start, t_end, num_steps, batch_size, dim, device ):
    """
    Initializes sampling from a gaussian distribution of given mean and std. Make sure that the 
    score function given accepts batched x [batch_size, dim] and t values of dim [batch_size].
    
    Returns: numpy array of time steps, and reverse process numpy array with dimension
    [time_step, batch_size, dim]
    """
    x = mean + std*torch.randn( [batch_size, dim], device=device)
    time_steps = torch.linspace(t_start, t_end, num_steps, device=device)
    dt = (t_start-t_end)/num_steps 
    
    track = [x.cpu().numpy(), ]
    
    for time_step in time_steps[1:]:
        
        f = x*drift(time_step)
        g = diffusion( torch.ones_like(x)*time_step )
        score = score_fn( torch.ones([batch_size], device=device)*time_step, x )
        
        if time_step > t_end:
            x_mean = x + ( g**2 * score - f ) * dt
            x = x_mean +  g * torch.randn_like(x) * dt**0.5
        
        if time_step == t_end:
            x =  x + ( g**2 * score - f ) * dt

        track.append( x.cpu().numpy() )
    
    return time_steps.cpu().numpy(),  np.array(track)


def reverse_sde_nn( mean, std, drift, diffusion, score_fn, pert_std, ema, t_start, t_end, num_steps, batch_size, dim, device ):
    """
    Initializes sampling from a gaussian distribution of given mean and std. Make sure that the 
    score function given accepts batched x [batch_size, dim] and t values of dim [batch_size].
    
    Returns: numpy array of time steps, and reverse process numpy array with dimension
    [time_step, batch_size, dim]
    """
    x = mean + std*torch.randn( [batch_size,dim], device=device )
    time_steps = torch.linspace(t_start, t_end, num_steps, device=device)
    dt = (t_start-t_end)/num_steps 
    
    track = [x.cpu().numpy(),]
    
    for time_step in time_steps[1:]:
        with torch.no_grad(), ema.average_parameters():
        
            f = x*drift(time_step)
            g = diffusion( torch.ones_like(x)*time_step )
            score = score_fn( torch.ones([batch_size], device=device)*time_step, x )/pert_std(time_step)

            if time_step > t_end:
                x_mean = x + ( g**2 * score - f ) * dt
                x = x_mean +  g * torch.randn_like(x) * dt**0.5

            if time_step==t_end:
                x =  x + ( g**2 * score - f ) * dt

        track.append( x.cpu().numpy() )
    
    return time_steps.cpu().numpy(),  np.array(track)


def sampling( mean, std, drift, diffusion, score_fn, pert_std, ema, t_start, t_end, num_steps, batch_size, dim, device ):
    """
    Initializes sampling from a gaussian distribution of given mean and std. Make sure that the 
    score function given accepts batched x [batch_size, dim] and t values of dim [batch_size].
    
    Returns: numpy array of time steps, and reverse process numpy array with dimension
    [time_step, batch_size, dim]
    """
    x = mean + std*torch.randn( [batch_size,dim], device=device )
    time_steps = torch.linspace(t_start, t_end, num_steps, device=device)
    dt = (t_start-t_end)/num_steps 
    
    for time_step in time_steps[1:]:
        with torch.no_grad(), ema.average_parameters():
        
            f = x*drift(time_step)
            g = diffusion( torch.ones_like(x)*time_step )
            score = score_fn( torch.ones([batch_size], device=device)*time_step, x )/pert_std(time_step)

            if time_step > t_end:
                x_mean = x + ( g**2 * score - f ) * dt
                x = x_mean +  g * torch.randn_like(x) * dt**0.5

            if time_step==t_end:
                x_final =  x + ( g**2 * score - f ) * dt
                return x_final
            
def sampling_true( mean, std, drift, diffusion, score_fn, t_start, t_end, num_steps, batch_size, dim, device ):
    """
    Initializes sampling from a gaussian distribution of given mean and std. Make sure that the 
    score function given accepts batched x [batch_size, dim] and t values of dim [batch_size].
    
    Returns: numpy array of time steps, and reverse process numpy array with dimension
    [time_step, batch_size, dim]
    """
    x = mean + std*torch.randn( [batch_size,dim], device=device )
    time_steps = torch.linspace(t_start, t_end, num_steps, device=device)
    dt = (t_start-t_end)/num_steps 
    
    for time_step in time_steps[1:]:
        
        f = x*drift(time_step)
        g = diffusion( torch.ones_like(x)*time_step )
        score = score_fn( torch.ones([batch_size], device=device)*time_step, x )

        if time_step > t_end:
            x_mean = x + ( g**2 * score - f ) * dt
            x = x_mean +  g * torch.randn_like(x) * dt**0.5

        if time_step==t_end:
            x_final =  x + ( g**2 * score - f ) * dt
            return x_final