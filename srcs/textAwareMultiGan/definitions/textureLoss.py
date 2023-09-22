import numpy as np
import cv2
import torch
import torch.nn as nn

class LBP:
    def calculateLBP(self, tensor):
        if tensor.dim() != 4:
            raise ValueError("Input tensor should be 4-dimensional (batch_size, channels, height, width)")

        filters = [
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]),
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),
            np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        ]

        batch_size, channels, height, width = tensor.shape
        convolved_images = []
        device = tensor.device

        for i in range(batch_size):
            for j in range(channels):
                imGray = tensor[i, j].cpu().numpy()
                convolved_images_channel = [cv2.filter2D(imGray, -1, f) for f in filters]

                for k in range(len(convolved_images_channel)):
                    convolved_images_channel[k][convolved_images_channel[k] >= 0] = 1
                    convolved_images_channel[k][convolved_images_channel[k] < 0] = 0

                convolved_image = np.uint8(np.sum([2 ** k * img for k, img in enumerate(convolved_images_channel)], axis=0))
                convolved_images.append(convolved_image / 255.0)  # Dividi per 255 per normalizzare

        # Aggiungi una dimensione del canale all'output
        return torch.unsqueeze(torch.tensor(np.array(convolved_images), dtype=torch.float32), dim=1)

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()

    def forward(self, real_image, fake_image):
        # Calcola LBP su real_image e fake_image
        lbp = LBP()
        lbp_input = lbp.calculateLBP(real_image)
        lbp_reference = lbp.calculateLBP(fake_image)

        # Calcola la differenza tra i risultati LBP
        diff = torch.abs(lbp_input - lbp_reference)

        # Calcola la norma L1 della differenza
        loss = torch.sum(diff)
        loss = torch.sqrt(loss)

        return loss
