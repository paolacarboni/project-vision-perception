import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, n):
        super(PatchGANDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, n, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(n, n * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(n * 2, n * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(n * 4, 1, kernel_size=4, stride=1, padding=1, bias=False))
        )

        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.05)


    def forward(self, x):
        return self.layers(x)

    def initialize_weights(self, std):
      print("Inizializzati")
      for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
          # Inizializza i pesi con una distribuzione Gaussiana
          torch.nn.init.normal_(m.weight.data, mean=0.0, std=std)
          if m.bias is not None:
            m.bias.data.zero_()
