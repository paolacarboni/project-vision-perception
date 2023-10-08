import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from ..definitions.utils import create_pyramid_image

class GanDataset(Dataset):
    def __init__(self, mode, img_path, real_path, file_selection=None):
        self.mode = mode
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.list_data = []
        self._load_data(img_path, real_path)
        #self.import_dataset()
    
    def set_device(self, device):
        self.device = device

    def _load_data(self, img_path, real_path):
        img_folder = os.listdir(img_path)
        real_folder = os.listdir(real_path)

        for input, real in tqdm(zip(img_folder, real_folder)):
            tensor_i = torch.load(os.path.join(img_folder, input))
            tensor_r = torch.load(os.path.join(real_folder, real))
            entry = {
                    'input': tensor_i,
                    'real': tensor_r
                }
            self.list_data.append(entry)       

    def _create_pyramid_images(self, img_path, mask_path, file_selection=None, shuffle=True):

        toTransform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        transform = transforms.Compose(toTransform)

        dataset = datasets.ImageFolder(root=img_path, transform=transform)
        
        toTransform.append(transforms.Grayscale(num_output_channels=1))
        masks = datasets.ImageFolder(root=mask_path, transform=transform)

        return dataset, masks


    def import_dataset(self):
        '''
        output:
        data = dictionary of images and masks loaded in DEVICE
        '''
        data = []
        m = 0

        for image, _ in tqdm(self.dataset):

            mask = self.masks[m][0]

            input = image * (1 - mask)

            final_input = torch.cat((input, mask), dim=0)

            entry = {
                'input': final_input,
                'real': image
            }
            self.list_data.append(entry)

            m += 1
            if m >= len(self.masks):
                m = 0
        return data
