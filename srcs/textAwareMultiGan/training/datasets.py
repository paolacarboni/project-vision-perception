import os
import math
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFilter

def change_ext(file_name, new_ext):

    base_name, ext = os.path.splitext(file_name)
    
    # Aggiungi la nuova estensione
    new_filename = base_name + new_ext
    
    return new_filename

def create_pyram_image(image, blur=False):
    result_image = Image.new('RGB', (512, 512))

    s1 = math.floor(image.size[0] / 256)
    s2 = math.floor(image.size[1] / 256)
    m = s1
    if s2 >= s1:
        m = s2
    if (blur and m > 1):
        image = image.filter(ImageFilter.GaussianBlur(radius=m))
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

    return result_image

def create_pyramids(old_path, new_path):
    texture_path = os.path.join(old_path, "textures")
    mask_path = os.path.join(old_path, "masks")
    folder_textures = os.listdir(texture_path)
    folder_masks = os.listdir(mask_path)

    toTransform = [
        transforms.ToTensor(),
    ]

    tran_text = transforms.Compose(toTransform)
    toTransformGrey = [
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ]
    tran_mask = transforms.Compose(toTransformGrey)

    m = 0

    for i in tqdm(folder_textures):
        try:
            image = Image.open(os.path.join(texture_path, i))
            text = create_pyram_image(image, blur=True)
            name = change_ext(i, ".png")
            text.save(os.path.join(new_path, 'reals', name))
            mask_image = Image.open(os.path.join(mask_path, folder_masks[m]))
            mask = create_pyram_image(mask_image)

            text = tran_text(text)
            mask = tran_mask(mask)

            input = text * (1 - mask)
            input = torch.cat((input, (1 - mask)), dim=0)

            input = transforms.ToPILImage()(input)
            input.save(os.path.join(new_path, 'inputs', name))
        except Exception as e:
            print("L'immagine che da problemi: ", os.path.join(texture_path, i))

        m += 1
        if m >= len(folder_masks):
            m = 0

def create_datasets(old_datasets_path, new_datasets_path, validation=False):

    os.makedirs(os.path.join(new_datasets_path,'train', 'inputs'))
    os.makedirs(os.path.join(new_datasets_path,'train', 'reals'))
    os.makedirs(os.path.join(new_datasets_path,'test', 'inputs'))
    os.makedirs(os.path.join(new_datasets_path,'test', 'reals'))

    try:
        create_pyramids(os.path.join(old_datasets_path, 'train'), os.path.join(new_datasets_path, 'train'))
        create_pyramids(os.path.join(old_datasets_path, 'test'), os.path.join(new_datasets_path, 'test'))
    except Exception as e:
        print(e)
        raise e
