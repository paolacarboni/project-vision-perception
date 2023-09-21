import os
from PIL import Image
import numpy as np
import torch
from torchvision import utils as vutils
from ..training.utils import epochCollector

def save_imgs(imgs, basename):
    try:
        i = 1
        for batch in imgs:
            filename = basename + '_e' + str(i) + '.jpg'
            vutils.save_image(batch.detach().cpu(), filename, nrow=8, normalize=True, pad_value=0.3)
            i += 1
    except Exception as e:
        error = Exception(f'Error: {__file__}: save_imgs: {str(e)}')
        raise error

def save_data(data: epochCollector, folder_path):
    
    lossname = "loss.npz"
    fakename = os.path.join(folder_path, "generated_grid")
    realname = os.path.join(folder_path, "original_grid")
    maskname =  os.path.join(folder_path, "maskered_grid")

    try:
        os.makedirs(folder_path, exist_ok=True)

        file_loss_name = os.path.join(folder_path, lossname)
        np.savez(file_loss_name, array1=data.get_discriminator_losses(), array2=data.get_generator_losses())
    except Exception as e:
        error = Exception(f"Error: {__file__}: save_data: {str(e)}")
        print(error)
        raise error
    try:
        save_imgs(data.fake_imgs, fakename)
        save_imgs(data.real_imgs, realname)
        save_imgs(data.mask_imgs, maskname)
    except Exception as e:
        print(e)
        raise e
