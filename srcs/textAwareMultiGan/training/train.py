import torch
import math
import os
from ..definitions.gan import GAN
from ..definitions.gan32 import GAN32
from ..definitions.gan64 import GAN64
from ..definitions.gan128 import GAN128
from ..definitions.gan256 import GAN256
from ..definitions.earlyStopper import EarlyStopper
from ..training.utils import trainParameters, epochCollector
from ..training.dataloader import make_dataloaders
from ..training.loader import load_parameters
from ..training.saver import save_data

def save(gan: GAN, save_folder, training_collector, test_collector, validation_collector=None):
    try:
        save_data(training_collector, os.path.join(save_folder, "train"))
        save_data(test_collector,  os.path.join(save_folder, "test"))
        if validation_collector != None:
            save_data(test_collector,  os.path.join(save_folder, "validation"))
        gan.save(os.path.join(save_folder, ("discriminator_" + str(gan.res) + ".pth")), os.path.join(save_folder, ("generator_" + str(gan.res) + ".pth")))
    except Exception as e:
        return 1


def train(gan: GAN, dataloaders = [], epoch=1, batch_size=32, validator: EarlyStopper=None, saving=0, save_folder=""):

    validation = validator != None
    if len(dataloaders) < 2 + int(validation):
        raise Exception("Wrong dataloader format")
    for dataload in dataloaders:
        if len(dataload) < 2:
            raise Exception("Wrong dataloader format")

    training_collector: epochCollector = epochCollector()
    test_collector : epochCollector = epochCollector()
    validation_collector = None
    if validation:
        validation_collector: epochCollector = epochCollector()

    for e in range(epoch):
        print('Start Train: e{}'.format(e))
        lossD_train, lossG_train, imgs_gen, imgs_real, imgs_mask = gan.exec_epoch(dataloaders[0][0], dataloaders[0][1], batch_size, True)
        print('End Train: e{} D(x)={:.4f} D(G(z))={:.4f}'.format(e, lossD_train, lossG_train))
        training_collector.append_losses(lossD_train, lossG_train)
        training_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

        print('Start Test: e{}'.format(e))
        lossD_test, lossG_test, imgs_gen, imgs_real, imgs_mask = gan.exec_epoch(dataloaders[1][0], dataloaders[1][1], batch_size, False)
        print('End Test:  e{} D(x)={:.4f} D(G(z))={:.4f}'.format(e, lossD_test, lossG_test))
        test_collector.append_losses(lossD_test, lossG_test)
        test_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

        if validation:
            print('Start Validation: e{}'.format(e))
            lossD_val, lossG_val, imgs_gen, imgs_real, imgs_mask = gan.exec_epoch(dataloaders[2][0], dataloaders[2][1], batch_size, False)
            print('End Validation:  e{} D(x)={:.4f} D(G(z))={:.4f}'.format(e, lossD_val, lossG_val))
            validation_collector.append_losses(lossD_val, lossG_val)
            validation_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)
            if validator.early_stop(lossG_val):             
                break
        if saving > 0 and (e % saving == 0):
            save(gan, save_folder, training_collector, test_collector, validation_collector)
    
    return training_collector, test_collector, validation_collector

def load(gan: GAN, par_file):

    if not gan.res or not gan.get_G():
        raise Exception("Generator not defined")

    res = int(math.log2(gan.res) - 4)

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

    i = 0
    for filename in to_load:
        gan.load_G_weights(filename, i)
        i += 1

    if par.d_filename:
        gan.load_D_weights(par.d_filename)

    gan.set_D_optimizer(torch.optim.Adam(gan.get_D().parameters(), lr=par.d_lr, betas=par.d_betas))
    gan.set_G_optimizer(torch.optim.Adam(gan.get_G().parameters(), lr=par.g_lr, betas=par.g_betas))

    return 0

def train_gan(res, datasets_path, save_folder, par_file, device, epoch=1, batch_size=32, validation=False, saving=0):

    if (res == 32):
        gan = GAN32(device)
    elif (res == 64):
        gan = GAN64(device)
    elif (res == 128):
        gan = GAN128(device)
    elif (res == 256):
        gan = GAN256(device)
    else:
        return 1

    validator = None
    if validation:
        validator: EarlyStopper = EarlyStopper(patience=3, min_delta=0.0005)

    if load(gan, par_file):
        return 1

    try:
        dataloaders = make_dataloaders(datasets_path, batch_size=batch_size, validation=validation)
    except Exception as e:
        print(str(e))
        return 1

    train_collector, test_collector, validation_collector = train(gan, dataloaders, epoch, batch_size, validator=validator, saving=saving, save_folder=save_folder)
    
    save(gan, save_folder, train_collector, test_collector, validation_collector)
    
    return 0
