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
    images_path_test = os.path.join(path, 'test')
    
    dataset = GanDataset(images_path_train)

    validation_split = 0.1

    dataset_size = len(dataset)
    validation_size = int(validation_split * dataset_size)
    train_size = dataset_size - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = GanDataset(images_path_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader

def train_gan(dataset_path, save_path, batch_size, resolution, epochs, lr_g=0.001, lr_d=0.001, betas_d=(0.5,0.99), betas_g=(0.5,0.99), patience=3, min_delta=0):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader, val_dataloader, test_dataloader = do_dataloader(batch_size, dataset_path)

    gan: TextAwareMultiGan = TextAwareMultiGan(resolution)
    gan.to(device)

    optimizer_d = (torch.optim.Adam(gan.get_discriminator().parameters(), lr=lr_d, betas=betas_d))
    optimizer_g = (torch.optim.Adam(gan.get_generator().parameters(), lr=lr_g, betas=betas_g))

    trainer: GanTrainer = GanTrainer(gan, optimizer_d, optimizer_g, "gan", patience=patience, min_delta=min_delta)
    trainer.to(device)

    train_d_losses, train_g_losses, valid_losses = trainer.train(batch_size, train_dataloader, val_dataloader, epochs=epochs)

    date = datetime.today().strftime('%Y-%m-%d')
    lossname = "loss_GAN_" + date
    d_name = "discriminator_" + date + '.pth'
    g_name = "generator_" + date + '.pth'

    np.savez(os.path.join(save_path, lossname), array1=train_d_losses, array2=train_g_losses, array3=valid_losses)
    gan.save(os.path.join(save_path, d_name), os.path.join(save_path, g_name))

