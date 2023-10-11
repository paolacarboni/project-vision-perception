import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from ..definitions.gan_final import TextAwareMultiGan
from ..definitions.datasets import GanDataset
from ..definitions.trainer import GanTrainer

def test_gan(gan: TextAwareMultiGan, dataset_path, batch_size: int = 32):
    
    dataset: GanDataset = GanDataset(dataset_path)
    test_loader: DataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    trainer: GanTrainer = GanTrainer(gan, None, None, "GAN")

    metrics, predictions = trainer.test(test_loader, batch_size)

    return metrics, predictions

def analysis(resolution, dataset_path, save_path, discriminator, generators=[], loss="", batch_size = 32):

    

    if loss != "":
        lossname = os.path.join(save_path, "loss_{}.png".format(pow(2, 0 + 5)))
        data = np.load(loss)

        lossD = data['array1']
        lossG = data['array2']
        validation = data['array3']

        plt.plot(lossD, label='LossD')
        plt.plot(lossG, label='LossG')
        plt.plot(validation, label='Validation')

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training {}'.format(pow(2, 0 + 5)))
        plt.grid(True)
        plt.savefig(lossname)
        exit()  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    gan: TextAwareMultiGan = TextAwareMultiGan(resolution)
    gan.to(device)

    for i in range(len(generators)):
        gan.load_G_weights(generators[i], i)
    gan.load_D(discriminator)

    metrics, predictions = test_gan(gan, dataset_path, batch_size)

    filename = os.path.join(save_path, "metrics.txt")
    img_min_m = os.path.join(save_path, "min_maskered.png")
    img_max_m = os.path.join(save_path, "max_maskered.png")
    img_min_p = os.path.join(save_path, "min_predicted.png")
    img_max_p = os.path.join(save_path, "max_predicted.png")
    img_min_o = os.path.join(save_path, "min_original.png")
    img_max_o = os.path.join(save_path, "max_original.png")

    with open(filename, "w") as file:
        file.write(
            f"Loss: {metrics['cross_entropy']},\nMSE: {metrics['MSE_entropy']},\nMin MSE: {predictions['min'][0]},\nMax MSE: {predictions['max'][0]}"
        )

    vutils.save_image(predictions['min'][1].detach().cpu(), img_min_m, nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(predictions['max'][1].detach().cpu(), img_max_m, nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(predictions['min'][2].detach().cpu(), img_min_p, nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(predictions['max'][2].detach().cpu(), img_max_p, nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(predictions['min'][3].detach().cpu(), img_min_o, nrow=8, normalize=True, pad_value=0.3)
    vutils.save_image(predictions['max'][3].detach().cpu(), img_max_o, nrow=8, normalize=True, pad_value=0.3)

    if loss != "":
        lossname = os.path.join(save_path, "loss_{}.png".format(pow(2, i + 5)))
        data = np.load(loss)

        lossD = data['array1']
        lossG = data['array2']
        #validation = data['array3']

        plt.plot(lossD, label='LossD')
        plt.plot(lossG, label='LossG')
        #plt.plot(validation, label='Validation')

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training {}'.format(pow(2, i + 5)))
        plt.grid(True)
        plt.savefig(lossname)
