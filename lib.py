import torch

def reparam_trick(z_mean, log_var):
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(std)
    return z_mean + std * eps


def vae_loss(x, x_hat, z_mean, log_var):
    bceloss = torch.nn.functional.binary_cross_entropy(x_hat, x.view(x.shape[0], -1), reduction='sum')
    kldloss = (1 + log_var - z_mean**2 - torch.exp(log_var)).sum() / 2
    return bceloss - kldloss