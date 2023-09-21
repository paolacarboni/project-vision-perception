import torch
import torch.nn as nn
from ..definitions.utils import load_weights, save_nn
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator

class GAN():
    def __init__(self, device):
        self.device = device
        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()
        self.D = PatchGANDiscriminator(3, 24).to(device)
        self.generators = []
        self.res = 0
        self.G
        self.optimizerD
        self.optimizerG

    def load_D_weights(self, filename):
        try:
            load_weights(self.D, filename)
            return 0
        except Exception as e:
            print(str(e))
            return 1

    def load_G_weights(self, filename, i=0):
        if len(self.generators) < i + 1:
            print("Error: generator not defined")
            return 1
        try:
            load_weights(self.generators[i], filename)
        except Exception as e:
            print(str(e))
            return 0

    def load(self, d_filename, g_filenames: []):
        if len(g_filenames) > len(self.generators):
            raise Exception("Error: wrong number of files")
        try:
            load_weights(self.D, d_filename)
            i = 0
            for f in g_filenames:
                load_weights(self.generators[i], f)
                i += 1
            print("Weights loaded")
            return 0
        except Exception as e:
            print(str(e))
            return 1

    def set_D_optimizer(self, optimizer):
        self.optimizerD = optimizer.to(self.device)

    def set_G_optimizer(self, optimizer):
        self.optimizerG = optimizer.to(self.device)

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
            if len(self.G):
                save_nn(self.G, g_filename)
            return 0
        except Exception as e:
            print(str(e))
            return 1