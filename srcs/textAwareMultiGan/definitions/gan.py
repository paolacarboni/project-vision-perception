import torch
import torch.nn as nn
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator

def load_weights(NN, filename, device):
    try:
        d_file = torch.load(filename, map_location=torch.device(device))
        NN.load_state_dict(d_file)
        return 0
    except FileNotFoundError:
        print(f"Error: {__file__}: load_weights: File '{filename}' doesn't exist.")
    except PermissionError:
        print(f"Error: {__file__}: load_weights: Permission denied: '{filename}'.")
    except Exception as e:
        print(f"Error: {__file__}: load_weights: Error: {e}")
    return 1

def save_nn(neuralN, filename):
    try:
        torch.save(neuralN.state_dict(), filename)
    except Exception as e:
        raise e

class GAN():
    def __init__(self, device):
        self.device = device
        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()
        self.D = PatchGANDiscriminator(3, 24).to(device)
        self.generators = []
        self.res = 0
        self.G = None
        self.optimizerD = None
        self.optimizerG = None

    def load_D_weights(self, filename):
        try:
            load_weights(self.D, filename, self.device)
            return 0
        except Exception as e:
            print(str(e))
            return 1

    def load_G_weights(self, filename, i=0):
        if len(self.generators) < i + 1:
            print("Error: generator not defined")
            return 1
        try:
            load_weights(self.generators[i], filename, self.device)
        except Exception as e:
            print(str(e))
            return 0

    def load(self, d_filename, g_filenames: []):
        if len(g_filenames) > len(self.generators):
            raise Exception("Error: wrong number of files")
        try:
            load_weights(self.D, d_filename, self.device)
            i = 0
            for f in g_filenames:
                load_weights(self.generators[i], f, self.device)
                i += 1
            print("Weights loaded")
            return 0
        except Exception as e:
            print(str(e))
            return 1

    def set_D_optimizer(self, optimizer):
        self.optimizerD = optimizer

    def set_G_optimizer(self, optimizer):
        self.optimizerG = optimizer

    def get_D(self):
        return self.D

    def get_G(self):
        return self.G

    def exec_epoch(self):
        if not self.optimizerD:
            raise Exception("OptimizerD not defined")
        if not self.optimizerG:
            raise Exception("OptimizerG not defined")

    def save(self, d_filename, g_filename):
        try:
            save_nn(self.D, d_filename)
            if self.G is not None:
                save_nn(self.G, g_filename)
            return 0
        except Exception as e:
            print(str(e))
            return 1