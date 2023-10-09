import torch
import torch.nn as nn
import torch.nn.init as init

class Generator128(nn.Module):
    def __init__(self):
        super(Generator128, self).__init__()

        # Blocco 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=28, out_channels=56, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Blocco 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=28, out_channels=56, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Blocco 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Blocco 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=168, out_channels=112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=112, out_channels=56, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=56, out_channels=28, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=28, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.initialize_weights()

    def forward(self, x_corr, x_64, x_32):
        out1 = self.block1(x_corr)
        #x_64_u = torch.nn.functional.interpolate(x_64, size=(128, 128), mode='bilinear', align_corners=False)
        out2 = self.block2(x_64)
        x_32_u = torch.nn.functional.interpolate(x_32, size=(64, 64), mode='bilinear', align_corners=False)
        out3 = self.block2(x_32_u)
        concatenated_features = torch.cat((out1, out2, out3), dim=1)
        out4 = self.block4(concatenated_features)
        return out4

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m in self.block1.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 1
                elif m in self.block2.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 2
                elif m in self.block3.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 3
                elif m in self.block4.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 4
                if m.bias is not None:
                    m.bias.data.zero_()
