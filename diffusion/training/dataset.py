import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Subset

def open_data( data_dir, filename, mlp=False ):
    if  "MNIST" in filename:
        transform = transforms.Compose([transforms.ToTensor()])
        all_mnist = MNIST(data_dir, train=True, transform=transform, download=False)
        
        integer_found = ''.join(char for char in filename if char.isdigit())
        if integer_found != '':
            indices = [i for i, (_, label) in enumerate(all_mnist) if label == int(integer_found)]
            dataset = [all_mnist[i][0] for i in indices] 
            dataset = torch.stack(dataset, dim=0)

        else:
            dataset = torch.stack([image for image, _ in all_mnist], dim=0)    

    if filename.endswith(".npy"):
        np_data = np.load( f'{data_dir}/{filename}' )
        dataset = torch.from_numpy( np_data )
        #if mlp==False:
        #    dataset = torch.unsqueeze( dataset, dim=1 )

    return dataset


class mult_datasets(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset1, *args, mlp=False):
        self.args = args
        self.datasets = {}
        self.datasets['dataset1'] = open_data( data_dir=data_dir, filename=dataset1, mlp=mlp )
        
        track = 2
        for arg in self.args:
            self.datasets[f'dataset{track}'] = open_data( data_dir=data_dir, filename=arg, mlp=mlp )
            
            if self.datasets[f'dataset{track}'].shape != self.datasets['dataset1'].shape:
                raise ValueError(f"Dataset {arg} does not have the same dimensions as dataset {dataset1}.")
            
            track+=1

    def __len__(self):
        return len( self.datasets['dataset1'] )

    def __getitem__(self, idx):
        data = []
        for i in range( 1, len(self.datasets)+1 ):
            data.append( self.datasets[f'dataset{i}'][idx] )

        return data
    
class dist_dataset(torch.utils.data.Dataset):
    def __init__(self, device, distribution, size):
        self.device = device
        self.distribution = distribution
        self.size = size

        self.data = self.distribution.sample([ self.size ])
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx].to(self.device)
    