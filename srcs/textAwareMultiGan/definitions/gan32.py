import torch
from ..definitions.generator32 import Generator32
from ..definitions.gan import GAN

class GAN32(GAN):
    def __init__(self, device):
        super().__init__(device)
        self.generators.append(Generator32().to(device))
        self.G = self.generators[0]
        self.res = 32

    def exec_epoch(self, dataloader_texture, dataloader_mask, batch_size=32, train=True):

        super().exec_epoch()

        D_32 = self.D
        G_32 = self.G
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG
        adversarial_loss = self.adversarial_loss
        pixelwise_loss = self.pixelwise_loss

        total_lossG = 0
        total_lossD = 0

        for i, data in enumerate(dataloader_texture, 0):
            x_mask, _ = next(iter(dataloader_mask))
            x_mask = x_mask[:, : , 0:32, 0:32].to(self.device)
            x_real, _ = next(iter(dataloader_texture))
            x_real = x_real[:, :, 0:32, 0:32].to(self.device)
            D_real = D_32(x_real)

            lab_real = torch.full((batch_size, 1, 3, 3), 0.99).to(self.device)
            lab_fake = torch.full((batch_size, 1, 3, 3), 0.01).to(self.device)
            
            if train:
                optimizerD.zero_grad()

            x_corr = x_real * (1 - x_mask)
            masked_images = []

            for image, mask in zip(x_corr, x_mask):
                masked_image = torch.cat((image, mask), dim=0)
                masked_images.append(masked_image)

            z = torch.stack(masked_images)

            x_gen = G_32(z)
            D_fake = D_32(x_gen)

            #print("D_fake, D_real: ", torch.sigmoid(D_real), torch.sigmoid(D_fake))
            lossD_real = adversarial_loss(torch.sigmoid(D_real), lab_real).to(self.device)
            lossD_fake = adversarial_loss(torch.sigmoid(D_fake), lab_fake).to(self.device)

            lossD = 0.5 * lossD_real + 0.5 * lossD_fake
            #print("LossD real: ", lossD_real.item())
            #print("Loss fake: ", lossD_fake.item())
            if train:
                lossD.mean().backward()
                optimizerD.step()
                optimizerG.zero_grad()

            x_gen = G_32(z)
            D_fake = D_32(x_gen)

            lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
            pixelwise_loss_value = pixelwise_loss(x_gen, x_real)

            #print("LossG_adv: ", lossG_adv.item())
            lossG = 0.1 * lossG_adv + pixelwise_loss_value

            if train:
                lossG.mean().backward()
                optimizerG.step()
            
            total_lossD += lossD.mean().item()
            total_lossG += lossG.mean().item()

            if i % 10 == 0:
                print('i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(i, len(dataloader_texture), lossD.mean().item(), lossG.mean().item()))

        total_lossD /= len(dataloader_texture)
        total_lossG /= len(dataloader_texture)

        return total_lossD, total_lossG, x_gen.detach().clone(), x_real.detach().clone(), x_corr.detach().clone()
    