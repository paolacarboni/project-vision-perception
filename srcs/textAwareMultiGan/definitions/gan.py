import math
import numpy as np
import torch
import os
from torchvision import utils as vutils
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from ..definitions.generator32 import Generator32 as G32
from ..definitions.generator64 import Generator64 as G64
from ..definitions.generator128 import Generator128 as G128
from ..definitions.generator256 import Generator256 as G256
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator as PD
from ..definitions.utils import create_pyramid_image

class TextAwareMultiGan():
    def __init__(self, res = 256):
        self.generators = [G32(), G64(), G128(), G256()]
        self.discriminators = [PD(3, 24), PD(3, 24), PD(3, 24), PD(3, 24)]
        self.set_resolution(res)

    def set_resolution(self, res):
        resolution = int(math.log(res, 2) - 4)
        if resolution not in [1, 2, 3, 4]:
            raise Exception("Error: bad number")
        self.resolution = resolution
        self.D = self.discriminators[resolution - 1]

    def get_resolution(self):
        return pow(2, self.resolution + 4)

    def get_generator(self):
        return self.generators[self.resolution - 1]

    def get_discriminator(self):
        return self.discriminators[self.resolution - 1]

    def load_G_weights(self, filename, i):
        if i >= 0 and i < 4:
            d_file = torch.load(filename, map_location=torch.device('cpu'))
            self.generators[i].load_state_dict(d_file)
    
    def load_D_weights(self, filename, i):
        if i >= 0 and i < 4:
            d_file = torch.load(filename, map_location=torch.device('cpu'))
            self.discriminators[i].load_state_dict(d_file)

    def load_D(self, filename):
        d_file = torch.load(filename, map_location=torch.device('cpu'))
        self.discriminators[self.resolution - 1].load_state_dict(d_file)

    def to(self, device):
        for gen in self.generators:
            gen.to(device)
        for dis in self.discriminators:
            dis.to(device)

    def save(self, d_name, g_name):
        d = self.get_discriminator()
        g = self.get_generator()

        d_name += '.pth'
        g_name += '.pth'

        torch.save(d.state_dict(), d_name)
        torch.save(g.state_dict(), g_name)

    def train(self):
        self.generators[self.resolution - 1].train()
        self.discriminators[self.resolution - 1].train()

    def eval(self):
        self.generators[self.resolution - 1].eval()
        self.discriminators[self.resolution - 1].eval()

    def train_generator(self):
        self.generators[self.resolution - 1].train()

    def train_discriminator(self):
        self.discriminators[self.resolution - 1].train()

    def eval_generator(self):
        self.generators[self.resolution - 1].eval()

    def eval_discriminator(self):
        self.discriminators[self.resolution - 1].eval()

    def pre_processing(self, image, mask):
        toTransform = [
            transforms.ToTensor(),
        ]

        tran_text = transforms.Compose(toTransform)
        toTransformGrey = [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ]
        tran_mask = transforms.Compose(toTransformGrey)
        
        i = create_pyramid_image(image, blur=True)
        m = create_pyramid_image(mask)

        text = tran_text(i)
        mask = tran_mask(m)

        input = text * (1 - mask)

        dim = input.dim() - 3
        if dim == 0:
            input = input.unsqueeze(0)
        input = torch.cat((input, (1 - mask)), dim=0)
        if dim == 0:
            input = input.squeeze(0)
        input = transforms.ToPILImage()(input)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        input = transform(input)

        return input

    def forward(self, batch):

        z_array = []
        for res in range(self.resolution):
            r = pow(2, res + 5)
            z_array.append(batch[..., (r-32):(2*r-32), 0:r])

        x = []

        for res in range(self.resolution):
            g = self.generators[res]
            z = z_array[res]
            x.insert(0, g(z, *x))

        return x[0]

    def __call__(self, inputs):
        return self.forward(inputs)
