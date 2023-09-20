import os
from PIL import Image
import numpy as np
from ..training.utils import epochCollector

def save_imgs(imgs, folder):
    try:
        m = 0
        n = 0
        for f in imgs:
            n = 0
            new_folder = os.path.join(folder, 'epoch_' + str(m))
            os.makedirs(new_folder, exist_ok=True)
            for i in f:
                image_path =  os.path.join(new_folder, 'image_' + str(n) + '.jpg')
                image = Image.fromarray(i)
                image.save(image_path)
                n+= 1
            m += 1
    except Exception as e:
        error = Exception(f'Error: {__file__}: save_imgs: {str(e)}')
        raise error

def save_data(data: epochCollector, folder_path):
    
    lossname = "loss.npz"
    fakefolder = "generated"
    realfolder = "original"
    maskfolder =  "maskered"

    try:
        os.makedirs(folder_path, exist_ok=True)

        file_loss_name = os.path.join(folder_path, lossname)
        np.savez(file_loss_name, array1=data.get_discriminator_losses(), array2=data.get_generator_losses())
        
        new_fake_folder = os.path.join(folder_path, fakefolder)
        os.makedirs(new_fake_folder, exist_ok=True)

        new_real_folder = os.path.join(folder_path, realfolder)
        os.makedirs(new_real_folder, exist_ok=True)

        new_mask_folder = os.path.join(folder_path, maskfolder)
        os.makedirs(new_mask_folder, exist_ok=True)
    except Exception as e:
        error = Exception(f"Error: {__file__}: save_data: {str(e)}")
        print(error)
        raise error
    try:
        save_imgs(data.fake_imgs, new_fake_folder)
        save_imgs(data.real_imgs, new_real_folder)
        save_imgs(data.mask_imgs, new_mask_folder)
    except Exception as e:
        print(e)
        raise e