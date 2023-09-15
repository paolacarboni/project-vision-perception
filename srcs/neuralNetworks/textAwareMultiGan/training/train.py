from srcs.neuralNetworks.textAwareMultiGan.training.datasets import make_dataloaders
import torch
from definitions.generator32 import Generator32
from definitions.patchGanDiscriminator import PatchGANDiscriminator
from training.utils import trainParameters
from training.train32 import train_32

def load_weights(D, G, d_filename, g_filename):
    if d_filename != "":
        try:
            d_file = open(d_filename, 'r')
            D.load_state_dict(d_file)
        except Exception as e:
            print(e)
    if g_filename != "":
        try:
            g_file = open(g_filename, 'r')
            G.load_state_dict(g_file)
        except Exception as e:
            print(e)


def train(par32: trainParameters, batch_size=32):

    D_32 = PatchGANDiscriminator(3, 24)
    G_32 = Generator32()
    optimizerD = torch.optim.Adam(D_32.parameters(), lr=par32.d_lr, betas=par32.d_betas)
    optimizerG = torch.optim.Adam(G_32.parameters(), lr=par32.g_lr, betas=par32.g_betas)

    load_weights(D_32, G_32, par32.d_filename, par32.g_filename)

    dl_train_text, dl_train_mask, dl_test_text, dl_test_mask = make_dataloaders(batch_size=batch_size)
    dataloaders = [[dl_train_text, dl_train_mask], [dl_test_text, dl_test_mask]]

    train_32(D_32, [G_32], optimizerD, optimizerG, dataloaders, batch_size=batch_size, num_epochs=par32.number_epochs)
