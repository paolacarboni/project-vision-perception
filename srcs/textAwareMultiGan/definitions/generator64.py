import torch
import torch.nn as nn
import torch.nn.init as init

class Generator64(nn.Module):
    def __init__(self):
        super(Generator64, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.initialize_weights()

    def forward(self, x_corr, x_32):
        out_block1 = self.block1(x_corr)
        #x_32_u = torch.nn.functional.interpolate(x_32, size=(64, 64), mode='bilinear', align_corners=False)
        out_block2 = self.block2(x_32)
        dim = x_corr.dim() - 3
        concatenated_features = torch.cat((out_block1, out_block2), dim=dim)
        out_block3 = self.block3(concatenated_features)
        return out_block3

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m in self.block1.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 1
                elif m in self.block2.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 2
                elif m in self.block3.modules():
                    init.normal_(m.weight.data, mean=0.0, std=0.02)  # Inizializza i pesi per il blocco 3
                if m.bias is not None:
                    m.bias.data.zero_()
