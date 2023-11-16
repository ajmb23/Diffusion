from diffusion.architectures.register_arch import register_model

import torch
import numpy as np
from torch import nn

@register_model(name='bb_MLP')
class bb_MLP(nn.Module):
    def __init__(self, dimensions, scale=16, embed_dim=8, units=100, **kwargs):
        super().__init__()

        self.hyperparameters = { "dimensions": dimensions }
        
        self.nn1 = nn.Linear(dimensions+embed_dim, units)
        self.nn2 = nn.Linear(units, units)
        self.nn3 = nn.Linear(units, dimensions)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.act = nn.SiLU()
    
    def forward(self, t, x, *cond):
        if cond:
            x = torch.cat([x , cond], dim=1)

        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        temb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        
        x = torch.cat([temb, x], dim=1)
        x = self.nn1(x)
        x = self.act(x)
        x = self.nn2(x)
        x = self.act(x)
        x = self.nn3(x)   
        return x