import os
import argparse
import torch
import torchvision
import models

import models
import lib

parser = argparse.ArgumentParser()
parser.add_argument("--img_type", type=str, default="MNIST", help="Type of image to train")
opt = parser.parse_args()

batch_size = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize([0.5],[0.5])
])

if opt.img_type == "MNIST":
    img_shape = (28, 28)
    TrainDataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)

    TestDataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            download=True,
                                            transform=transform)

elif opt.img_type == "CIFAR10":
    img_shape = (32, 32)
    TrainDataset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, 
                                                download=True, 
                                                transform=transform)
                                            
    TestDataset = torchvision.datasets.CIFAR10(root='./data', 
                                               train=False, 
                                               download=True,
                                               transform=transform)


TrainDataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
TestDataLoader = torch.utils.data.DataLoader(TestDataset, batch_size=batch_size, shuffle=False, num_workers=2)


epochs = 400
net = models.VAE()
bce = torch.nn.BCELoss()
optim = torch.optim.Adam(net.parameters())

if torch.cuda.is_available():
    net = net.cuda()
    bce = bce.cuda()

for epoch in range(epochs):
    net.train()
    train_loss = 0.0
    for (x, _) in TrainDataLoader:
        num_data = x.shape[0]
        if torch.cuda.is_available():
            x = x.cuda()

        optim.zero_grad()
        output, z_mean, log_var = net.forward(x)
        
        loss = lib.vae_loss(x, output, z_mean, log_var)
        #print(loss)
        loss.backward()
        train_loss += loss.item()
        optim.step()

    print("[Epoch %d/%d] Train Loss: %.3f"%(epoch+1, epochs, train_loss / len(TrainDataLoader.dataset)))

    net.eval()
    loss = 0.0
    with torch.no_grad():
        for (x, _) in TestDataLoader:
            num_data = x.shape[0]
            if torch.cuda.is_available():
                x = x.cuda()

            _x = x.view(num_data, -1)
            output, z_mean, log_var = net.forward(x)
            loss += lib.vae_loss(_x, output, z_mean, log_var).sum()

        loss /= len(TestDataLoader.dataset)
        print("\tTest Loss: %.3f"%(loss))

    if epoch % 10 == 9:
        path = "./imgs"
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path + "/%03d.png"%(epoch+1)
        with torch.no_grad():
            sample = torch.randn(64, 20)
            if torch.cuda.is_available():
                sample = sample.cuda()

            x_g = net.decode(sample).view(64, -1, *img_shape)
            torchvision.utils.save_image(x_g, path)
