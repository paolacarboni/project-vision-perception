import torch
from ..definitions.generator32 import Generator32
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator
from ..training.utils import trainParameters
from ..training.train32 import train_32
from ..training.dataloader import make_dataloaders
from ..training.loader import load_parameters, load_weights
from ..training.saver import save_data

def train_gan(res, datasets_path, save_folder, par_file, device, epoch=1, batch_size=32):

    par32: trainParameters = trainParameters()
    par64: trainParameters = trainParameters()
    par128: trainParameters = trainParameters()
    par256: trainParameters = trainParameters()

    if load_parameters([par32, par64, par128, par256], par_file):
        return(1)

    D_32 = PatchGANDiscriminator(3, 24).to(device)
    G_32 = Generator32().to(device)

    if par32.d_filename and load_weights(D_32, par32.d_filename):
        return(1)
    if par32.g_filename and load_weights(G_32, par32.g_filename):
        return(1)

    try:
        dl_train_text, dl_train_mask, dl_test_text, dl_test_mask = make_dataloaders(datasets_path, batch_size=batch_size)
    except Exception as e:
        return 1
    dataloaders = [[dl_train_text, dl_train_mask], [dl_test_text, dl_test_mask]]

    if res == 32:
        optimizerD = torch.optim.Adam(D_32.parameters(), lr=par32.d_lr, betas=par32.d_betas)
        optimizerG = torch.optim.Adam(G_32.parameters(), lr=par32.g_lr, betas=par32.g_betas)
        training_collector, test_collector = train_32(D_32, [G_32], optimizerD, optimizerG, dataloaders, batch_size=batch_size, num_epochs=epoch, device=device)
        try:
            save_data(training_collector, save_folder + "/train")
            save_data(test_collector, save_folder + "/test")
        except Exception as e:
            return 1
    
    return 0
