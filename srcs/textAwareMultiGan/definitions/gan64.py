import torch
import torch.nn.functional as F
from ..definitions.gan import GAN
from ..definitions.generator32 import Generator32
from ..definitions.generator64 import Generator64

class GAN64(GAN):
    def __init__(self, device):
        super().__init__(device)
        self.generators.append(Generator32().to(device))
        self.generators.append(Generator64().to(device))
        self.G = self.generators[1]
        self.res = 64
    
    def exec_epoch(self, dataloader_texture, dataloader_mask, batch_size=32, train=True):
        super().exec_epoch()

        D_64 = self.D
        G_32 = self.generators[0]
        G_64 = self.G
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG
        adversarial_loss = self.adversarial_loss
        pixelwise_loss = self.pixelwise_loss

        total_lossG = 0
        total_lossD = 0

        for i, data in enumerate(dataloader_texture, 0):
            x_mask, _ = next(iter(dataloader_mask))
            x_mask = x_mask.to(self.device)
            x_real, _ = next(iter(dataloader_texture))
            x_real = x_real.to(self.device)
            x_real_64 = x_real[:, :, 32:96, 0:64].to(self.device)

            D_real = D_64(x_real_64)

            lab_real = torch.full((batch_size, 1, 7, 7), 0.9, device=self.device)
            lab_fake = torch.full((batch_size, 1, 7, 7), 0.1, device=self.device)
            
            if train:
                optimizerD.zero_grad()

            x_corr = x_real * (1 - x_mask)
            masked_images = []

            for image, mask in zip(x_corr, x_mask):
                masked_image = torch.cat((image, mask), dim=0)
                masked_images.append(masked_image)

            z = torch.stack(masked_images)

            x_gen_32 = G_32(z[:, :, 0:32, 0:32])
            x_gen = G_64(z[:, :, 32:96, 0:64], x_gen_32)
            D_fake = D_64(x_gen)

            lossD_real = adversarial_loss(torch.sigmoid(D_real), lab_real)
            lossD_fake = adversarial_loss(torch.sigmoid(D_fake), lab_fake)

            lossD = lossD_real + lossD_fake

            if train:
                lossD.backward()
                optimizerD.step()
                optimizerG.zero_grad()

            x_gen_32 = G_32(z[:, :, 0:32, 0:32])
            x_gen = G_64(z[:, :, 32:96, 0:64], x_gen_32)
            D_fake = D_64(x_gen)

            lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
            pixelwise_loss_value = pixelwise_loss(x_gen, x_real_64)

            lossG = 0.1 * lossG_adv + pixelwise_loss_value

            if train:
                lossG.backward()
                optimizerG.step()

            total_lossD += lossD.mean().item()
            total_lossG += lossG.mean().item()

            if i % 10 == 0:
                print('i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(i, len(dataloader_texture), lossD.mean().item(), lossG.mean().item()))

        total_lossD /= len(dataloader_texture)
        total_lossG /= len(dataloader_texture)

        return total_lossD, total_lossG, x_gen.detach().clone(), x_real_64.detach().clone(), x_corr[:, :, 32:96, 0:64].detach().clone()

    def forward(self, image, mask):

        i_32 = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
        m_32 = F.interpolate(mask, size=(32, 32), mode='bilinear', align_corners=False)
        
        i_64 = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
        m_64 = F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)


        x_32 = torch.cat((i_32 * (1 - m_32), m_32), dim=1)
        x_gen_32 = self.generators[0](x_32)

        x_64 = torch.cat((i_64 * (1 - m_64), m_64), dim=1)
        x_gen = self.G(x_64, x_gen_32)
        return x_gen
