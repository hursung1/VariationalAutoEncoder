import torch
import torchvision

import lib

class VAE(torch.nn.Module):
    def __init__(self, input_data_shape = (28, 28),  hidden_layer_num = 400, latent_dim = 20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_layer_num = input_data_shape[0] * input_data_shape[1]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_layer_num, hidden_layer_num),
            torch.nn.ReLU()
        )

        self.z_mean_out = torch.nn.Linear(hidden_layer_num, latent_dim)
        self.log_var_out = torch.nn.Linear(hidden_layer_num, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_layer_num),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden_layer_num, self.input_layer_num),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z_mean, log_var = self.encode(x)
        latent_variable = lib.reparam_trick(z_mean, log_var)
        return self.decode(latent_variable), z_mean, log_var
    
    def encode(self, x):
        _x = x.view(-1, 28*28)
        out = self.encoder(_x)
        z_mean = self.z_mean_out(out)
        log_var = self.log_var_out(out)
        return z_mean, log_var

    def decode(self, x):
        return self.decoder(x)
