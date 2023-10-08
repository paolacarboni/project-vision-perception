import math
import numpy as np
import torch
from PIL import Image
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

    def to(self, device):
        for gen in self.generators:
            gen.to(device)
        for dis in self.discriminators:
            dis.to(device)

    def save(self, d_name, g_name):
        d = self.get_discriminator()
        g = self.get_generator()

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

    def pre_processing(image, mask):
        i = create_pyramid_image(image)
        m = create_pyramid_image(mask)

        input = i * (1 - m)
        mask_array = np.array(m)
        input_array = np.array(input)

        final_input = np.concatenate((input_array, mask_array), axis=2)
        final_input = Image.fromarray(final_input)

        return final_input

    def forward(self, batch):

        z_array = []
        start = 0
        for res in range(self.resolution):
            r = self.get_resolution()
            z_array.append(batch[..., start:(r+start), 0:r])
            start += r

        x = []

        for res in range(self.resolution):
            g = self.generators[res]
            z = z_array[res]
            x.append(g(z, *x))
        
        return x[-1]

    def __call__(self, inputs):
        return self.forward(inputs)
