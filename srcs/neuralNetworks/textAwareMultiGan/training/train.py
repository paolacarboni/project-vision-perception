import ast
import numpy as np
import torch
import os
import tkinter as tk
from PIL import Image

from tkinter import filedialog
from ..definitions.generator32 import Generator32
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator
from ..training.utils import trainParameters, epochCollector
from ..training.train32 import train_32
from ..training.datasets import make_dataloaders

PARAMETERS_FILE = os.path.join(os.path.dirname(__file__), 'parameters.txt')

def load_weights(NN, filename):
    try:
        d_file = open(filename, 'r')
        NN.load_state_dict(d_file)
        return 0
    except FileNotFoundError:
        print(f"File '{PARAMETERS_FILE}' doesn't exist.")
    except PermissionError:
        print(f"Permission denied: '{PARAMETERS_FILE}'.")
    except Exception as e:
        print(f"Error: {e}")
    return 1

def parse_line(line, par: trainParameters):
    if len(line) > 1:
        parameter = line[1]
        if line[0] in ["d_lr", "g_lr"]:
            try:
                parameter = float(line[1])
            except ValueError:
                print(f'{line[1]} isn\'t a float.')
                return 1
        if line[0] in ["d_betas", "g_betas"]:
            try:
                parameter = ast.literal_eval(line[1])
            except ValueError:
                print(f'{line[1]} isn\'t a tuple.')
                return 1
        if line[0] == "d_lr":
            par.d_lr = parameter
        elif line[0] == "g_lr":
            par.g_lr = parameter
        elif line[0] == "d_betas":
            par.d_betas = parameter
        elif line[0] == "g_betas":
            par.g_betas = parameter
        elif line[0] == "d_filename":
            par.d_filename = parameter
        elif line[0] == "g_filename":
            par.g_filename = parameter
    return 0

def load_parameters(par32: trainParameters, par64: trainParameters, par128: trainParameters, par256: trainParameters):
    
    par: trainParameters

    try:
        file = open(PARAMETERS_FILE, 'r')
        for line in file:
            words = line.split()
            if len(words) > 0:
                if words[0] == "32":
                    par = par32
                elif words[0] == "64":
                    par = par64
                elif words[0] == "128":
                    par = par128
                elif words[0] == "256":
                    par = par256
                else:
                    if parse_line(words, par):
                        return 1
        return 0
    except FileNotFoundError:
        print(f"File '{PARAMETERS_FILE}' doesn't exist.")
    except PermissionError:
        print(f"Permission denied: '{PARAMETERS_FILE}'.")
    except Exception as e:
        print(f"Error: {e}")
    return 1

def save_imgs(imgs, folder):
    m = 0
    n = 0
    for f in imgs:
        n = 0
        new_folder = os.path.join(folder, 'epoch_' + str(m))
        os.makedirs(new_folder, exist_ok=True)
        for i in f:
            image_path =  os.path.join(new_folder, 'image_' + str(n) + '.jpg')
            image = Image.fromarray(i)
            image.save(image_path)
            n+= 1
        m += 1
    pass

def save_data(data: epochCollector, lossname, fakefolder, realfolder, maskfolder):

    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()

    if folder_path:
        pass
    else:
        folder_path = r'C:\Users\jacop\Desktop\Studio\VisionPercepion\project-vision-perception\resrcs\results'
    
    file_loss_name = os.path.join(folder_path, lossname)
    np.savez(file_loss_name, array1=data.get_discriminator_losses(), array2=data.get_generator_losses())
    
    new_fake_folder = os.path.join(folder_path, fakefolder)
    os.makedirs(new_fake_folder, exist_ok=True)

    new_real_folder = os.path.join(folder_path, realfolder)
    os.makedirs(new_real_folder, exist_ok=True)

    new_mask_folder = os.path.join(folder_path, maskfolder)
    os.makedirs(new_mask_folder, exist_ok=True)

    save_imgs(data.fake_imgs, new_fake_folder)
    save_imgs(data.real_imgs, new_real_folder)
    save_imgs(data.mask_imgs, new_mask_folder)

def train_gan(res, datasets_path, epoch=1, batch_size=32):

    par32: trainParameters = trainParameters()
    par64: trainParameters = trainParameters()
    par128: trainParameters = trainParameters()
    par256: trainParameters = trainParameters()

    if load_parameters(par32, par64, par128, par256):
        return(1)

    D_32 = PatchGANDiscriminator(3, 24)
    G_32 = Generator32()

    if par32.d_filename and load_weights(D_32, par32.d_filename):
        return(1)
    if par32.g_filename and load_weights(G_32, par32.g_filename):
        return(1)

    try:
        dl_train_text, dl_train_mask, dl_test_text, dl_test_mask = make_dataloaders(datasets_path, batch_size=batch_size)
    except Exception as e:
        print(e)
        return 1
    dataloaders = [[dl_train_text, dl_train_mask], [dl_test_text, dl_test_mask]]

    if res == 32:
        optimizerD = torch.optim.Adam(D_32.parameters(), lr=par32.d_lr, betas=par32.d_betas)
        optimizerG = torch.optim.Adam(G_32.parameters(), lr=par32.g_lr, betas=par32.g_betas)
        training_collector, test_collector = train_32(D_32, [G_32], optimizerD, optimizerG, dataloaders, batch_size=batch_size, num_epochs=epoch)
        save_data(training_collector, "train_loss.npz", "train_fake", "train_real", "train_mask")
        save_data(test_collector, "test_loss.npz", "test_fake", "test_real", "test_mask")
    
    return 0
