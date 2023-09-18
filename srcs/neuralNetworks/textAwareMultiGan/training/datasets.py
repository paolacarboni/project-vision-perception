import torch
from torchvision import transforms
from torchvision import datasets
from PIL import Image, ImageFilter
import os

PARAMETERS_FILE = os.path.join(os.path.dirname(__file__), 'parameters.txt')

train_textures_path = os.path.join(os.path.dirname(__file__),r'../../../../resrcs/datasets/textAwareMultiGan/training/textures')
train_masks_path = os.path.join(os.path.dirname(__file__),r'../../../../resrcs/datasets/textAwareMultiGan/training/masks')
test_textures_path = os.path.join(os.path.dirname(__file__),r'../../../../resrcs/datasets/textAwareMultiGan/test/textures')
test_masks_path = os.path.join(os.path.dirname(__file__),r'../../../../resrcs/datasets/textAwareMultiGan/test/masks')

new_train_textures_path = os.path.join(os.path.dirname(__file__), r'../../../datasets/textAwareMultiGan/training/textures')
new_train_masks_path = os.path.join(os.path.dirname(__file__), r'../../../datasets/textAwareMultiGan/training/masks')
new_test_textures_path = os.path.join(os.path.dirname(__file__), r'../../../datasets/textAwareMultiGan/test/textures')
new_test_masks_path = os.path.join(os.path.dirname(__file__), r'../../../datasets/textAwareMultiGan/test/masks')

def do_dataloader(folder: str, batch_size=32, shuffle=True, tran = []):

    toTransform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    for i in tran:
        toTransform.append(i)
    transform = transforms.Compose(toTransform)

    path = folder
    try:
        dataset = datasets.ImageFolder(root=path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except Exception as e:
        print("Error: srcs/neuralNetworks/textAwareMultiGan/training/datasets.py: do_dataloader: ", e)
        exit(1)

    return dataloader

#Questa funzione crea i dataloader per il train e il test
def make_dataloaders(batch_size=32, shuffle=True):
    dl_train_text = do_dataloader(new_train_textures_path, batch_size=batch_size, shuffle=shuffle)
    dl_train_mask = do_dataloader(new_train_masks_path, batch_size=batch_size, shuffle=shuffle, tran = [transforms.Grayscale(num_output_channels=1)])
    dl_test_text = do_dataloader(new_test_textures_path, batch_size=batch_size, shuffle=shuffle)
    dl_test_mask = do_dataloader(new_test_masks_path, batch_size=batch_size, shuffle=shuffle, tran = [transforms.Grayscale(num_output_channels=1)])
    return dl_train_text, dl_train_mask, dl_test_text, dl_test_mask


#Questa funzione resize ogni immagine presente nella cartella oldpath in 4 immagini di dimensioni
#32x32, 64x64, 128x128 e 512x512 e le incolla in un'immagine 512x512 che salva nella cartella new_path.
#Se il parametro blur è True prima di fare il resize è applicato un filtro gaussiano all'immagine
def create_pyramid_images(old_path, new_path, blur=False):
    if os.path.exists(new_path):
        for file in os.listdir(new_path):
            complete_path = os.path.join(new_path, file)
            os.remove(complete_path)
    else:
        print("Error: srcs/neuralNetworks/textAwareMultiGan/training/datasets.py: create_pyramid_images: Folder not found: {}", new_path)
        exit(0)

    if os.path.exists(old_path):
        folder = os.listdir(old_path)
        for i in folder:
            width, height = 512, 512
            result_image = Image.new('RGB', (width, height))

            image = Image.open(old_path + '/' + i)
            image_256 = image.resize((256, 256))
            result_image.paste(image_256, (0, 224))

            if (blur):
                image_256 = image_256.filter(ImageFilter.GaussianBlur(radius=2.0))
            image_128 = image_256.resize((128, 128))
            result_image.paste(image_128, (0, 96))

            if (blur):
                image_128 = image_128.filter(ImageFilter.GaussianBlur(radius=2.0))
            image_64 = image_128.resize((64, 64))
            result_image.paste(image_64, (0, 32))

            if (blur):
                image_64 = image_64.filter(ImageFilter.GaussianBlur(radius=2.0))
            image_32 = image_64.resize((32, 32))
            result_image.paste(image_32, (0, 0))

            result_image.save(new_path + '/' + i, 'PNG')
    else:
        print("Folder not found: {}", old_path)


def create_datasets():
    create_pyramid_images(train_textures_path, new_train_textures_path + '/data', True)
    create_pyramid_images(train_masks_path, new_train_masks_path + '/data')
    create_pyramid_images(test_textures_path, new_test_textures_path + '/data', True)
    create_pyramid_images(test_masks_path, new_test_masks_path + '/data')
