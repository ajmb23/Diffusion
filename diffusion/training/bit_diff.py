import torch

def int_to_bits(x, bits):
    """
    Takes in a batch of tensor x, of integer values and converts them
    to bits based on the number of bits indicated in the channels component
    of the original x. 
    """
    B, C, *D = x.shape
    
    x = x.int()
    x = x.unsqueeze(dim=2)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device)
    mask = mask.reshape( -1, *[1]*len(D) )

    bits = ((x & mask) != 0).float()
    bits = bits.reshape( B, -1, *D )
    bits = bits * 2 - 1

    return bits