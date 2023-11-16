import torch 
import numpy as np 
import torch.distributions as td


def gaussians(mean, std, weight, device):
    """
    Returns multimodal gaussians the dimension is set by mean and std
    mean: torch tensor of floats for mean of different modes dimension 
    std: torch tensor of floats for std of different modes
    weight: scaling factor of weight of each mode 
    """
    mix = td.Categorical(weight.to(device), validate_args=False)  
    comp =  td.Independent( td.Normal( loc=mean.to(device), scale=std.to(device), validate_args=False ), 1 )
    return td.MixtureSameFamily(mix, comp, validate_args=False)

def two_moons(device, modes=128, width=0.1, size=1 ):
    """
    Returns a 2 moons distributions from a mixture of <modes> gaussian distributions
    :param modes: Number of modes inside each moon
    :param width: Width of the moons
    :param size: scales the coordinates by this amount
    """
    outer_circ_x = torch.cos(torch.linspace(0, np.pi, modes)) - .5
    outer_circ_y = torch.sin(torch.linspace(0, np.pi, modes)) - .25
    inner_circ_x = - torch.cos(torch.linspace(0, np.pi, modes)) + .5
    inner_circ_y = - torch.sin(torch.linspace(0, np.pi, modes)) + .25
    x = torch.concat([outer_circ_x, inner_circ_x])
    y = torch.concat([outer_circ_y, inner_circ_y])
    coords = size * torch.stack([x, y], dim=1).to(device)
    mixture = td.Categorical(probs=torch.ones(2*modes).to(device), validate_args=False)  # Uniform
    component = td.Independent(td.Normal(loc=coords, scale=width, validate_args=False), 1)  # Diagonal Multivariate Normal
    return td.MixtureSameFamily(mixture, component, validate_args=False)