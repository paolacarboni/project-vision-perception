import torch
import torch.nn as nn
import torch.nn.init as init

class Generator32(nn.Module):
    def __init__(self):
        super(Generator32, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def initialize_weights(self):
      print("Inizializzati")
      for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
          # Inizializza i pesi con una distribuzione Gaussiana
          init.normal_(m.weight.data, mean=0.0, std=0.02)
          if m.bias is not None:
            m.bias.data.zero_()
