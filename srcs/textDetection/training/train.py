import os 
import torch
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from ..definitions.losses import TextLoss
from ..definitions.trainer import UNetTrainer
from ..definitions.datasets import UNetDataset
from ..definitions.uNetTextDetection import UNetTextDetection

def check_not_path(path):
    return not(os.path.isdir(path))
    

def build_dataloader(path, batch_size):
    '''
    Datasets loader: from path load Train Dataset, Validation Dataset, Test Dataset
    '''

    images_path_train = os.path.join(path, 'train\images')
    masks_path_train = os.path.join(path, 'train\masks')

    images_path_validation = os.path.join(path, 'validation\images')
    masks_path_validation = os.path.join(path, 'validation\masks')

    images_path_test = os.path.join(path, 'test\images')
    masks_path_test = os.path.join(path, 'test\masks')

    print("\nReading Train Set...")
    train_dataset = UNetDataset("train", images_path_train, masks_path_train)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    print("\nReading Val Set...")
    val_dataset = UNetDataset("dev", images_path_validation, masks_path_validation)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    print("\nReading Test Set...")
    test_dataset = UNetDataset("test", images_path_test, masks_path_test)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, val_dataloader, test_dataloader

def train_text(batch_size, epochs, dataset_path, save_path, lr=1e-5, weight_decay=1e-3, patience=3, min_delta=0):

    if check_not_path(dataset_path):
        print("Error: dataset_path isn't a valid path")
        return 1
    elif check_not_path(save_path):
        print("Error: save_path isn't a valid path")
        return 1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Dataset loader
    train_dataloader, val_dataloader, test_dataloader = build_dataloader(dataset_path, batch_size)

    # Model definition and training 
    model = UNetTextDetection().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = TextLoss()
    trainer = UNetTrainer(model, optimizer, loss, 'chk', patience, min_delta)
    train_losses, validation_losses = trainer.train(train_dataloader, val_dataloader, epochs)

    # Save model
    trainer.save_model(save_path, 'textDetection')
    print('Model saved in', save_path)

    # Save lossses
    date = datetime.today().strftime('%Y-%m-%d')
    loss_file = os.path.join(save_path, "saved_losses_" + date + ".npz");
    epochs = np.array(range(len(train_losses))) + 1
    np.savez(loss_file, epochs = epochs, train_losses = np.array(train_losses), validation_losses = np.array(validation_losses))
    print('Losses saved in', loss_file)

    
