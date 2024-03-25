import torch
from diffusion.training import int_to_bits

def one_epoch( device, dataloader, score_model, optimizer, ema, pert_mshift, 
               pert_std, min_t, max_t, grad_clip=0, cond_noise=0, bit=False ):

    data_iter = iter(dataloader)
    sum_loss_iter = torch.tensor([0.0], device=device)
    counter = torch.tensor([0], dtype=torch.int32, device=device)

    for i in range( len(dataloader) ):

        X = next(data_iter)
        if isinstance(X, list):
            x, *args = X
        else:
            x = X
            args = []

        if bit:
            x = int_to_bits(x=x, bits=bit )

        B, *D = x.shape

        x = x.to(device)
        for i, tensor in enumerate(args):
            args[i] = tensor.to(device)
        
        if cond_noise>0:
            cond_n = cond_noise * torch.randn_like(x)
            args = [tensor + cond_n for tensor in args]
        
        t = torch.empty( B ).uniform_( min_t , max_t ).to( device )
        mean_shift = pert_mshift(t).reshape(B, *[1]*len(D))
        std = pert_std(t).reshape(B, *[1]*len(D))
        z = torch.randn_like(x) 
        perturbed_x = x*mean_shift + std*z
        
        optimizer.zero_grad()
        model_out = score_model(t, perturbed_x, *args)
        loss = torch.sum(torch.square(model_out + z)) / B
        
        loss.backward()
        if grad_clip > 0:     
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), grad_clip)
        optimizer.step()
        ema.update(score_model.parameters())

        sum_loss_iter += loss
        counter += 1

    return sum_loss_iter, counter