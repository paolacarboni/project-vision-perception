import torch
from torchvision import transforms
from ..definitions.gan import GAN
from ..definitions.generator32 import Generator32
from ..definitions.generator64 import Generator64
from ..definitions.generator128 import Generator128
from ..definitions.textureLoss import TextureLoss

class GAN256(GAN):
    def __init__(self, device):
        super().__init__(device)
        self.generators.append()
        self.generators.append()
        self.generators.append()
        self.generators.append()
        self.G = self.generators[3]
        self.res = 256
        self.texture_loss = TextureLoss()
    
    def exec_epoch(self, dataloader_texture, dataloader_mask, batch_size=32, train=True):
        super().exec_epoch()

        D_256 = self.D
        G_32 = self.generators[0]
        G_64 = self.generators[1]
        G_128 = self.generators[2]
        G_256 = self.G
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG

        adversarial_loss = self.adversarial_loss
        pixelwise_loss = self.pixelwise_loss
        texture_loss = self.texture_loss

        total_lossG = 0
        total_lossD = 0

        for i, data in enumerate(dataloader_texture, 0):
            x_mask, _ = next(iter(dataloader_mask))
            x_mask = x_mask.to(self.device)
            x_real, _ = next(iter(dataloader_texture))
            x_real = x_real.to(self.device)
            x_real_256 = x_real[:, :, 224:480, 0:256]

            D_real = D_256(x_real_256)

            lab_real = torch.full((batch_size, 1, 31, 31), 0.9, device=self.device)
            lab_fake = torch.full((batch_size, 1, 31, 31), 0.1, device=self.device)
            
            if train:
                optimizerD.zero_grad()

            x_corr = x_real * (1 - x_mask)
            masked_images = []

            for image, mask in zip(x_corr, x_mask):
                masked_image = torch.cat((image, mask), dim=0)
                masked_images.append(masked_image)

            z = torch.stack(masked_images)

            x_gen_32 = G_32(z[:, :, 0:32, 0:32])
            x_gen_64 = G_64(z[:, :, 32:96, 0:64], x_gen_32)
            x_gen_128 = G_128(z[:, :, 96:224, 0:128], x_gen_64, x_gen_32)
            x_gen = G_256(z[:, :, 224:480, 0:256], x_gen_128, x_gen_64, x_gen_32)
            D_fake = D_256(x_gen)

            lossD_real = adversarial_loss(torch.sigmoid(D_real), lab_real)
            lossD_fake = adversarial_loss(torch.sigmoid(D_fake), lab_fake)

            lossD = lossD_real + lossD_fake

            if train:
                lossD.backward()
                optimizerD.step()
                optimizerG.zero_grad()

            x_gen_32 = G_32(z[:, :, 0:32, 0:32])
            x_gen_64 = G_64(z[:, :, 32:96, 0:64], x_gen_32)
            x_gen_128 = G_128(z[:, :, 96:224, 0:128], x_gen_64, x_gen_32)
            x_gen = G_256(z[:, :, 224:480, 0:256], x_gen_128, x_gen_64, x_gen_32)
            D_fake = D_256(x_gen)

            lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
            pixelwise_loss_value = pixelwise_loss(x_gen, x_real_256)

            grayscale_converter = transforms.Grayscale(num_output_channels=1)

            x_g_gen = torch.stack([grayscale_converter(image) for image in x_gen])
            x_g_gen = x_g_gen.to(self.device)

            x_g_real = torch.stack([grayscale_converter(image) for image in x_real_256])
            x_g_real = x_g_real.to(self.device)

            texture_loss_value = texture_loss(x_g_real, x_g_gen.detach())
            loss_g1 = (1 / 1000) * texture_loss_value
            lossG = 0.1 * lossG_adv + (0.7 * pixelwise_loss_value) + ((0.3 / 1000) * texture_loss_value)
            #lossG = 0.1 * lossG_adv + pixelwise_loss_value + 10 * texture_loss_value

            if train:
                lossG.backward()
                optimizerG.step()

            total_lossD += lossD.mean().item()
            total_lossG += lossG.mean().item()

            if i % 10 == 0:
                print('i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(i, len(dataloader_texture), lossD.mean().item(), lossG.mean().item()))

        total_lossD /= len(dataloader_texture)
        total_lossG /= len(dataloader_texture)

        return total_lossD, total_lossG, x_gen.detach().clone(), x_real_256.detach().clone(), x_corr.detach().clone()
