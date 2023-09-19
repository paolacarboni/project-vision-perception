class epochCollector:

    def __init__(self):
        self.fake_imgs = []
        self.real_imgs = []
        self.mask_imgs = []
        self.losses = []
        self.losses.append([])
        self.losses.append([])

    def append_fake_img(self, img):
        self.fake_imgs.append(img)

    def append_real_img(self, img):
        self.real_imgs.append(img)
    
    def append_mask_img(self, img):
        self.mask_imgs.append(img)

    def append_imgs(self, fake, real, mask):
        self.append_fake_img(fake)
        self.append_real_img(real)
        self.append_mask_img(mask)

    def append_losses(self, d_loss, g_loss):
        self.losses[0].append(d_loss)
        self.losses[1].append(g_loss)

    def get_discriminator_losses(self):
        return self.losses[0]

    def get_generator_losses(self):
        return self.losses[1]


class trainParameters:

    def __init__(self):
        self.d_lr = 0.0
        self.g_lr = 0.0
        self.d_betas = 0.0
        self.g_betas = 0.0
        self.d_filename = ""
        self.g_filename = ""
        self.number_epochs = 0