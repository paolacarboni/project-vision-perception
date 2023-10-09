import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GanDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'inputs')
        self.real_dir = os.path.join(root_dir, 'reals')
        self.input_images = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.real_images = [f for f in os.listdir(self.real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        real_image_path = os.path.join(self.real_dir, self.real_images[idx])
        
        input_image = Image.open(input_image_path).convert('RGBA')
        real_image = Image.open(real_image_path).convert('RGB')
        
        input_image = self.transform(input_image)
        real_image = self.transform(real_image)
        
        return {'inputs': input_image, 'reals': real_image}