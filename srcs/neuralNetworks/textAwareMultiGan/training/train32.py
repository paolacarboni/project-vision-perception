import torch
import neuralNetworks as nn
from asyncio.windows_events import NULL
from definitions.patchGanDiscriminator import PatchGANDiscriminator
from definitions.generator32 import Generator32
from training.utils import epochCollector

adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

def epoch(D_32, G_32, optimizerD, optimizerG, dataloader_texture, dataloader_mask, batch_size=32, train=True):
    for i, data in enumerate(dataloader_texture, 0):
        x_mask, _ = next(iter(dataloader_mask))
        x_mask = x_mask[:, : , 0:32, 0:32]
        x_real, _ = next(iter(dataloader_texture))
        x_real = x_real[:, :, 0:32, 0:32]
        D_real = D_32(x_real)

        lab_real = torch.full((batch_size, 1, 3, 3), 0.9)
        lab_fake = torch.full((batch_size, 1, 3, 3), 0.1)
        
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

        lossD_real = adversarial_loss(torch.sigmoid(D_real), lab_real)
        lossD_fake = adversarial_loss(torch.sigmoid(D_fake), lab_fake)

        lossD = lossD_real + lossD_fake

        if train:
            lossD.backward()
            optimizerD.step()
            optimizerG.zero_grad()

        x_gen = G_32(z)
        D_fake = D_32(x_gen)

        lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
        pixelwise_loss_value = pixelwise_loss(x_gen, x_real)

        lossG = 0.1 * lossG_adv + pixelwise_loss_value

        if train:
            lossG.backward()
            optimizerG.step()

    return lossD.mean().item(), lossG.mean().item(), x_gen.detach().clone(), x_real.detach().clone(), x_corr.detach().clone()


def train_32(discriminator, generators, optimizerD, optimizerG, dataloaders, num_epochs=10, batch_size=32):

    D_32: PatchGANDiscriminator = discriminator
    G_32: Generator32 = generators[0]

    training_collector: epochCollector = epochCollector()
    test_collector : epochCollector = epochCollector()

    for epoch in range(num_epochs):
        lossD_train, lossG_train, imgs_gen, imgs_real, imgs_mask = epoch(G_32, D_32, optimizerG, optimizerD, dataloaders[0][0], dataloaders[0][1], train=True, batch_size=batch_size)
        print('Train: e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_train, lossG_train))
        training_collector.append_losses(lossD_train, lossG_train)
        training_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

        lossD_test, lossG_test, imgs_gen, imgs_real, imgs_mask = epoch(G_32, D_32, optimizerG, optimizerD, dataloaders[1][0], dataloaders[1][1], train=False, batch_size=batch_size)
        print('Test:  e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_test, lossG_test))
        test_collector.append_losses(lossD_test, lossG_test)
        test_collector.append_imgs(imgs_gen, imgs_real, imgs_mask)

    return training_collector, test_collector
