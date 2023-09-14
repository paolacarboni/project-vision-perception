import torch
import neuralNetworks as nn
from asyncio.windows_events import NULL
from definitions.patchGanDiscriminator import PatchGANDiscriminator
from definitions.generator32 import Generator32

adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

def test_epoch(G_32, D_32, dataloader_texture, dataloader_mask, device):
  for i, data in enumerate(dataloader_texture, 0):
    x_mask, _ = next(iter(dataloader_mask))
    x_mask = x_mask.to(device)
    x_mask = x_mask[:, : , 0:32, 0:32]
    x_real, _ = next(iter(dataloader_texture))
    x_real = x_real.to(device)
    x_real = x_real[:, :, 0:32, 0:32]

    lab_real = torch.full((32, 1, 3, 3), 0.9, device=device)
    lab_fake = torch.full((32, 1, 3, 3), 0.1, device=device)

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

    lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
    pixelwise_loss_value = pixelwise_loss(x_gen, x_real)

    lossG = 0.1 * lossG_adv + pixelwise_loss_value

  return lossG.mean().item()


def train_epoch(G_32, D_32, optimizerG, optimizerD, dataloader_texture, dataloader_mask, device):
    for i, data in enumerate(dataloader_texture, 0):
        x_mask, _ = next(iter(dataloader_mask))
        x_mask = x_mask.to(device)
        x_mask = x_mask[:, : , 0:32, 0:32]
        x_real, _ = next(iter(dataloader_texture))
        x_real = x_real.to(device)
        x_real = x_real[:, :, 0:32, 0:32]
        D_real = D_32(x_real)

        lab_real = torch.full((32, 1, 3, 3), 0.9, device=device)
        lab_fake = torch.full((32, 1, 3, 3), 0.1, device=device)
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

        lossD.backward()
        optimizerD.step()

        optimizerG.zero_grad()

        x_gen = G_32(z)
        D_fake = D_32(x_gen)

        lossG_adv = adversarial_loss(torch.sigmoid(D_fake),  lab_real)
        pixelwise_loss_value = pixelwise_loss(x_gen, x_real)

        lossG = 0.1 * lossG_adv + pixelwise_loss_value

        lossG.backward()
        optimizerG.step()
    return lossD.mean().item, lossG.mean().item()


def train_32(dataloader_texture, dataloader_mask, device, filenameG="", filenameD="", lrG=0.0005, lrD=0.0001, betasG=(0.5, 0.99), betasD=(0.5, 0.99), num_epochs=10):

    D_32 = PatchGANDiscriminator(3, 24).to(device)
    G_32 = Generator32().to(device)

    optimizerG = torch.optim.Adam(G_32.parameters(), lr=lrG, betas=betasG)
    optimizerD = torch.optim.Adam(D_32.parameters(), lr=lrD, betas=betasD)

    if (filenameG != ""):
        try:
            g32_file = torch.load(filenameG, map_location=torch.device(device))
            G_32.load_state_dict(g32_file)
        except Exception as e:
            return e
    if (filenameD != ""):
        try:
            d32_file = torch.load(filenameD, map_location=torch.device(device))
            D_32.load_state_dict(d32_file)
        except Exception as e:
            return e
    
    for epoch in range(num_epochs):
        lossD_train, lossG_train = train_epoch(G_32, D_32, optimizerG, optimizerD, device, dataloader_texture, dataloader_mask)
        print('e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_train, lossG_train))
        lossD_test, lossG_test = test_epoch(G_32, D_32, dataloader_texture, dataloader_mask, device)
        print('e{} D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, lossD_test, lossG_test))


    return NULL