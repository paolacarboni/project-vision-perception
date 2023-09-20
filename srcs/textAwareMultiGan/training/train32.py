import torch
import torch.nn as nn
from ..definitions.patchGanDiscriminator import PatchGANDiscriminator
from ..definitions.generator32 import Generator32
from ..training.utils import epochCollector

adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

def exec_epoch(D_32, G_32, optimizerD, optimizerG, dataloader_texture, dataloader_mask, device, batch_size=32, train=True):
    
    total_lossG = 0
    total_lossD = 0
    
    for i, data in enumerate(dataloader_texture, 0):
        x_mask, _ = next(iter(dataloader_mask))
        x_mask = x_mask[:, : , 0:32, 0:32].to(device)
        x_real, _ = next(iter(dataloader_texture))
        x_real = x_real[:, :, 0:32, 0:32].to(device)
        D_real = D_32(x_real)

        lab_real = torch.full((batch_size, 1, 3, 3), 0.9).to(device)
        lab_fake = torch.full((batch_size, 1, 3, 3), 0.1).to(device)
        
        if train:
            optimizerD.zero_grad()

        x_corr = x_real * (1 - x_mask)
        masked_images = []

        for image, mask in zip(x_corr, x_mask):
            masked_image = torch.cat((image, mask), dim=0)
            masked_images.append(masked_image)

        z = torch.stack(masked_images).to(device)

        x_gen = G_32(z)
        D_fake = D_32(x_gen)

        lossD_real = adversarial_loss(torch.sigmoid(D_real), lab_real)
        lossD_fake = adversarial_loss(torch.sigmoid(D_fake), lab_fake)

        lossD = lossD_real + lossD_fake

        if train:
            lossD.mean().backward()
            optimizerD.step()
            optimizerG.zero_grad()

        x_gen = G_32(z)
        D_fake = D_32(x_gen)

        lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
        pixelwise_loss_value = pixelwise_loss(x_gen, x_real)

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


def train_32(discriminator, generators, optimizerD, optimizerG, dataloaders, device, num_epochs=10, batch_size=32):

    D_32: PatchGANDiscriminator = discriminator
    G_32: Generator32 = generators[0]

    training_collector: epochCollector = epochCollector()
    test_collector : epochCollector = epochCollector()

    for epoch in range(num_epochs):
        lossD_train, lossG_train, imgs_gen, imgs_real, imgs_mask = exec_epoch(D_32, G_32, optimizerD, optimizerG, dataloaders[0][0], dataloaders[0][1], device, train=True, batch_size=batch_size)
        print('Train: e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_train, lossG_train))
        training_collector.append_losses(lossD_train, lossG_train)
        training_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

        lossD_test, lossG_test, imgs_gen, imgs_real, imgs_mask = exec_epoch(D_32, G_32, optimizerD, optimizerG, dataloaders[1][0], dataloaders[1][1], device, train=False, batch_size=batch_size)
        print('Test:  e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_test, lossG_test))
        test_collector.append_losses(lossD_test, lossG_test)
        test_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

    return training_collector, test_collector
