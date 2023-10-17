import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from torchvision.transforms.functional import to_pil_image
from torchmetrics.image import StructuralSimilarityIndexMeasure

class GanTester():
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cpu')
        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()

    def mean_absolute_error(self, y_true, y_pred):
        mae = F.l1_loss(y_pred, y_true)
        return mae.item()

    def structural_similarity(self, y_true, y_pred):
        ssim = StructuralSimilarityIndexMeasure(dat_range=1.0)
        return ssim(y_true, y_pred)

    def root_mean_square_error(self, image1, image2):
        squared_diff = (image1 - image2).pow(2).mean()
        rmse = torch.sqrt(squared_diff)
        return rmse.item()

    def test(self, test_dataset, o_batch_size):
        self.model.eval()
        metrics = {}
        predictions = {
            "max":[float('-inf'), None, None, None],
            "min":[float('inf'), None, None, None]
        }
        binary_losses = 0
        MSE_losses = 0
        MAE_losses = 0
        SSIM_losses = 0
        RMSE_losses = 0
        with torch.no_grad():
            batch_pbar = tqdm(test_dataset, desc = "Test - Batch", leave = False)
            for batch in batch_pbar:
                input = batch['inputs'].to(self.device)
                real = batch['reals'].to(self.device)
                batch_size = len(input)
                r = self.model.get_resolution()
                real_b = real[..., (r-32):((2*r)-32), 0:r]
                input_b = input[..., (r-32):((2*r)-32), 0:r]
                D = self.model.D
                
                size = int((self.model.get_resolution() / 8 - 1))
                lab_real = torch.full((batch_size, 1, size, size), 0.9).to(self.device)
                
                prediction = self.model(input)

                D_fake = D(prediction)

                lossG_adv = self.adversarial_loss(torch.sigmoid(D_fake),  lab_real)
                pixelwise_loss_value = self.pixelwise_loss(prediction, real_b)
                lossG = 0.1 * lossG_adv + pixelwise_loss_value
                
                binary_losses += lossG.item()

                MSE_loss = F.mse_loss(prediction, real_b)
                MSE_losses += MSE_loss.item()

                MAE_loss = self.mean_absolute_error(real_b, prediction)
                MAE_losses += MAE_loss

                SSIM_loss = self.structural_similarity(real_b, prediction)
                SSIM_losses += SSIM_loss

                RMSE_loss = self.root_mean_square_error(real_b, prediction)
                RMSE_losses += RMSE_loss

                if batch_size == o_batch_size:
                    if MSE_loss > predictions['max'][0]:
                        predictions['max'] = [MSE_loss, input_b, prediction, real_b]
                    if MSE_loss < predictions['min'][0]:
                        predictions['min'] = [MSE_loss, input_b, prediction, real_b]

        metrics['LOSS'] = binary_losses / len(test_dataset)
        metrics['MSE'] = MSE_losses / len(test_dataset)
        metrics['MAE'] = MAE_losses / len(test_dataset)
        metrics['SSIM'] = SSIM_losses / len(test_dataset)
        metrics['RMSE'] = RMSE_losses / len(test_dataset)

        return metrics, predictions
