import torch
import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from ..definitions.gan_final import TextAwareMultiGan
from ..definitions.datasets import GanDataset
from ..definitions.trainer import GanTrainer

def do_dataloader(batch_size, path):

    images_path_train = os.path.join(path, 'train')
    
    dataset = GanDataset(images_path_train)

    validation_split = 0.1

    dataset_size = len(dataset)
    validation_size = int(validation_split * dataset_size)
    train_size = dataset_size - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader

def train_gan(dataset_path, save_path, batch_size, resolution, epochs, generators = [], lr_g=0.001, lr_d=0.001, betas_d=(0.5,0.99), betas_g=(0.5,0.99), patience=3, min_delta=0.02):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader, val_dataloader = do_dataloader(batch_size, dataset_path)

    gan: TextAwareMultiGan = TextAwareMultiGan(resolution)
    gan.to(device)

    for i in range(len(generators)):
        gan.load_G_weights(generators[i], i)

    optimizer_d = (torch.optim.Adam(gan.get_discriminator().parameters(), lr=lr_d, betas=betas_d))
    optimizer_g = (torch.optim.Adam(gan.get_generator().parameters(), lr=lr_g, betas=betas_g))

    trainer: GanTrainer = GanTrainer(gan, optimizer_d, optimizer_g, "gan", patience=patience, min_delta=min_delta)
    trainer.to(device)

    train_d_losses, train_g_losses, valid_losses = trainer.train(train_dataloader, val_dataloader, save_path, epochs=epochs)

    lossname = "loss_GAN"
    d_name = "discriminator_" + str(gan.get_resolution())
    g_name = "generator_" + str(gan.get_resolution())

    np.savez(os.path.join(save_path, lossname), array1=train_d_losses, array2=train_g_losses, array3=valid_losses)
    gan.save(os.path.join(save_path, d_name), os.path.join(save_path, g_name))
