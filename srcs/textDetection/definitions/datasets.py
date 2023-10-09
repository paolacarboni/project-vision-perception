import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class UNetDataset(Dataset):
    def __init__(self, mode, img_path, mask_path, file_selection=None):
        self.mode = mode
        self.dataset, self.masks = self.read_dataset(img_path, mask_path, file_selection)
        self.list_data = self.import_dataset()


    def read_dataset(self, img_path, mask_path, file_selection=None):

        '''
        function to get images and masks from folders

        input:
            img_path = path where photos are stored
            mask_path = path where masks are stored (.png)
            file_selection = a list of elements in img_path,
                in such a way as to import only a selection of the elements present in the folder
        NOTE: masks must have the same name as the corresponding images

        output:
            dataset = list of images with height and width 256, format PIL.Image
            masks = list of masks with height and width 256, format PIL.Image
        '''

        dataset = []
        masks = []

        if img_path[-1]!='/':
            img_path = img_path + '/'
        if mask_path[-1]!='/':
            mask_path = mask_path + '/'

        if file_selection:
          photo_paths = file_selection
        else:
          photo_paths = os.listdir(img_path)

        from_img_to_tensor = transforms.ToTensor()
        for f in photo_paths:
            img = Image.open(img_path+ f)
            img = img.resize((256, 256))


            resized_img = from_img_to_tensor(img)
            resized_img = (resized_img-torch.mean(resized_img))/torch.std(resized_img)
            dataset.append(resized_img)

            mask = Image.open(mask_path+ f[:-3]+'png')
            mask = mask.resize((256, 256))
            mask = from_img_to_tensor(mask)
            masks.append(mask)
        return dataset, masks

    def import_dataset(self, device='cpu'):
        '''
        output:
        data = dictionary of images and masks loaded in DEVICE
        '''
        data = []
        for input, mask in zip(self.dataset, self.masks):
            entry = {
                'input': input.to(device),
                'mask': mask.to(device)
            }
            data.append(entry)
        return data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        return self.list_data[idx]

    def __str__(self):
        out = "set type: " + self.mode + "Set"
        out += "\nset length: " + str(len(self))
        return out