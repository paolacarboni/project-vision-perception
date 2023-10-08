import math
from PIL import Image, ImageFilter

#Questa funzione resize ogni immagine presente nella cartella oldpath in 4 immagini di dimensioni
#32x32, 64x64, 128x128 e 512x512 e le incolla in un'immagine 512x512 che salva nella cartella new_path.
#Se il parametro blur è True prima di fare il resize è applicato un filtro gaussiano all'immagine
def create_pyramid_image(image, blur=False):

    width, height = 512, 512

    result_image = Image.new('RGB', (width, height))

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
