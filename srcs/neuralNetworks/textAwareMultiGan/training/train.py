import torch
import os
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
        if line[0] in ["d_lr", "g_lr", "d_betas", "g_betas"]:
            try:
                parameter = float(line[1])
            except ValueError:
                print(f'{line[1]} isn\'t a float.')
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
                    if parse_line(line, par):
                        return 1
        return 0
    except FileNotFoundError:
        print(f"File '{PARAMETERS_FILE}' doesn't exist.")
    except PermissionError:
        print(f"Permission denied: '{PARAMETERS_FILE}'.")
    except Exception as e:
        print(f"Error: {e}")
    return 1

def save_data(data: epochCollector):
    pass

def train_gan(res, epoch=1, batch_size=32):

    par32: trainParameters = trainParameters()
    par64: trainParameters = trainParameters()
    par128: trainParameters = trainParameters()
    par256: trainParameters = trainParameters()

    if load_parameters(par32, par64, par128, par256):
        exit(1)

    D_32 = PatchGANDiscriminator(3, 24)
    G_32 = Generator32()

    if par32.d_filename and load_weights(D_32, par32.d_filename):
        exit (1)
    if par32.g_filename and load_weights(G_32, par32.g_filename):
        exit(1)

    dl_train_text, dl_train_mask, dl_test_text, dl_test_mask = make_dataloaders(batch_size=batch_size)
    dataloaders = [[dl_train_text, dl_train_mask], [dl_test_text, dl_test_mask]]

    if res == 32:
        optimizerD = torch.optim.Adam(D_32.parameters(), lr=par32.d_lr, betas=par32.d_betas)
        optimizerG = torch.optim.Adam(G_32.parameters(), lr=par32.g_lr, betas=par32.g_betas)
        training_collector, test_collector = train_32(D_32, [G_32], optimizerD, optimizerG, dataloaders, batch_size=batch_size, num_epochs=epoch)
        save_data(training_collector)
    
