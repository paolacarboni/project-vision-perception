import torch
import math
import os
from ..definitions.gan import GAN
from ..definitions.gan32 import GAN32
from ..training.utils import trainParameters, epochCollector
from ..training.dataloader import make_dataloaders
from ..training.loader import load_parameters
from ..training.saver import save_data

def save(gan: GAN, training_collector, test_collector, save_folder):
    try:
        save_data(training_collector, os.path.join(save_folder, "train"))
        save_data(test_collector,  os.path.join(save_folder, "test"))
        gan.save(os.path.join(save_folder, ("discriminator_" + str(gan.res) + ".pth")), os.path.join(save_folder, ("generator_" + str(gan.res) + ".pth")))
    except Exception as e:
        return 1


def train(gan: GAN, dataloaders = [], epoch=1, batch_size=32):

    if len(dataloaders) < 2 or (len(dataloaders[0]) < 2 or len(dataloaders[1]) < 2):
        raise Exception("Wrong dataloader format")

    training_collector: epochCollector = epochCollector()
    test_collector : epochCollector = epochCollector()

    for e in range(epoch):
        lossD_train, lossG_train, imgs_gen, imgs_real, imgs_mask = gan.exec_epoch(dataloaders[0][0], dataloaders[0][1], batch_size, True)
        print('Train: e{} D(x)={:.4f} D(G(z))={:.4f}'.format(e, lossD_train, lossG_train))
        training_collector.append_losses(lossD_train, lossG_train)
        training_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

        lossD_test, lossG_test, imgs_gen, imgs_real, imgs_mask = gan.exec_epoch(dataloaders[1][0], dataloaders[1][1], batch_size, False)
        print('Test:  e{} D(x)={:.4f} D(G(z))={:.4f}'.format(e, lossD_test, lossG_test))
        test_collector.append_losses(lossD_test, lossG_test)
        test_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)
    
    return training_collector, test_collector

def load(gan: GAN, par_file):

    if not gan.res or not gan.get_G():
        raise Exception("Generator not defined")

    res = math.log2(gan.res) - 4

    parameters = [
        trainParameters(),
        trainParameters(),
        trainParameters(),
        trainParameters()
    ]

    if load_parameters(parameters, par_file):
        return(1)

    par: trainParameters = parameters[res - 1]

    i = 0
    to_load = []
    while i < len(parameters) and parameters[i].g_filename:
        to_load.append(parameters[i].g_filename)
        i += 1

    if par.d_filename:
        gan.load(par.d_filename, to_load)

    gan.set_D_optimizer(torch.optim.Adam(gan.get_D().parameters(), lr=par.d_lr, betas=par.d_betas))
    gan.set_G_optimizer(torch.optim.Adam(gan.get_G().parameters(), lr=par.g_lr, betas=par.g_betas))

    return 0

def train_gan(res, datasets_path, save_folder, par_file, device, epoch=1, batch_size=32):

    if (res == 32):
        gan = GAN32(device)
    elif (res == 64):
        pass
    elif (res == 128):
        pass
    elif (res == 256):
        pass
    else:
        return 1

    if load(gan, par_file):
        return 1

    try:
        dl_train_text, dl_train_mask, dl_test_text, dl_test_mask = make_dataloaders(datasets_path, batch_size=batch_size)
    except Exception as e:
        print(str(e))
        return 1
    dataloaders = [[dl_train_text, dl_train_mask], [dl_test_text, dl_test_mask]]

    training_collector, test_collector = train(gan, dataloaders, epoch, batch_size)
    
    save(gan, training_collector, test_collector, save_folder)
    
    return 0
