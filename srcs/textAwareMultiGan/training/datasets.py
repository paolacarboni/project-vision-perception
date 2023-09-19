import os
from PIL import Image, ImageFilter


#Questa funzione resize ogni immagine presente nella cartella oldpath in 4 immagini di dimensioni
#32x32, 64x64, 128x128 e 512x512 e le incolla in un'immagine 512x512 che salva nella cartella new_path.
#Se il parametro blur è True prima di fare il resize è applicato un filtro gaussiano all'immagine
def create_pyramid_images(old_path, new_path, blur=False):
    if os.path.exists(new_path):
        for file in os.listdir(new_path):
            complete_path = os.path.join(new_path, file)
            os.remove(complete_path)
    else:
        os.makedirs(new_path)

    if os.path.exists(old_path):
        folder = os.listdir(old_path)
        for i in folder:
            width, height = 512, 512

            try:
                image = Image.open(old_path + '/' + i)
                result_image = Image.new('RGB', (width, height))

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
            except Exception as e:
                pass
    else:
        error = f'Error: {__file__}: create_pyrimid_images: Folder not found: {old_path}'
        raise error


def create_datasets(old_datasets_path, new_datasets_path):

    train_textures_path = old_datasets_path + '/training/textures'
    train_masks_path = old_datasets_path + '/training/masks'
    test_textures_path = old_datasets_path + '/test/textures'
    test_masks_path = old_datasets_path + '/training/masks'

    new_train_textures_path = new_datasets_path + '/training/textures'
    new_train_masks_path = new_datasets_path + '/training/masks'
    new_test_textures_path = new_datasets_path + '/test/textures'
    new_test_masks_path = new_datasets_path + '/training/masks'

    try:
        create_pyramid_images(train_textures_path, new_train_textures_path + '/data', True)
        create_pyramid_images(train_masks_path, new_train_masks_path + '/data')
        create_pyramid_images(test_textures_path, new_test_textures_path + '/data', True)
        create_pyramid_images(test_masks_path, new_test_masks_path + '/data')
    except Exception as e:
        print(e)
        raise e
