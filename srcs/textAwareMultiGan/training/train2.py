import torch
import os
import numpy as np
from datetime import datetime
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from ..definitions.gan_final import TextAwareMultiGan
from ..definitions.datasets import GanDataset
from ..definitions.trainer import GanTrainer

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'inputs')
        self.real_dir = os.path.join(root_dir, 'reals')
        self.input_images = os.listdir(self.input_dir)
        self.real_images = os.listdir(self.real_dir)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        real_image_path = os.path.join(self.real_dir, self.real_images[idx])
        
        input_image = Image.open(input_image_path).convert('RGBA')
        real_image = Image.open(real_image_path).convert('RGB')
        
        input_image = self.transform(input_image)
        real_image = self.transform(real_image)
        
        return {'inputs': input_image, 'reals': real_image}

def build_dataloader(batch_size, path):

    images_path_train = os.path.join(path, r'train\textures')
    masks_path_train = os.path.join(path, r'train\masks')

    #images_path_validation = os.path.join(path, 'validation\images')
    #masks_path_validation = os.path.join(path, 'validation\masks')

    images_path_test = os.path.join(path, r'test\textures')
    masks_path_test = os.path.join(path, r'test\masks')


    print("\nReading Train Set...")
    t_dataset = GanDataset("train", images_path_train, masks_path_train)
    
    num_samples = len(t_dataset)
    train_size = int(0.9 * num_samples)
    test_size = num_samples - train_size

    train_dataset, val_dataset = random_split(t_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)

    print("\nReading Test Set...")
    test_dataset = GanDataset("test", images_path_test, masks_path_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)

    return train_dataloader, val_dataloader, test_dataloader

def do_dataloader(batch_size, path, device):
    images_path_train = os.path.join(path, 'train')

    images_path_test = os.path.join(path, 'test')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    dataset = CustomDataset(images_path_train)
    #dataset = ImageFolder(root=images_path_train, transform=transform)

    validation_split = 0.1  # Ad esempio, il 20% dei dati sarà destinato alla validazione

    # Calcola le dimensioni dei set di addestramento e validazione
    dataset_size = len(dataset)
    validation_size = int(validation_split * dataset_size)
    train_size = dataset_size - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ImageFolder(root=images_path_test, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader

def train_gan(dataset_path, save_path, batch_size, resolution, epochs, lr_g=0.001, lr_d=0.001, betas_d=(0.5,0.99), betas_g=(0.5,0.99), patience=3, min_delta=0):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader, val_dataloader, test_dataloader = do_dataloader(batch_size, dataset_path, device)

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

