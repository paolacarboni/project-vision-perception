import os
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from tqdm.auto import tqdm

class GanTrainer():
    def __init__(self, model, optimizer_d, optimizer_g, chk_name, patience=1, min_delta=0):
        self.model = model
        self.d_optimizer = optimizer_d
        self.g_optimizer = optimizer_g
        self.chk_name = chk_name
        self.patience = patience
        self.min_validation_loss = float('inf')
        self.min_delta = min_delta
        self.counter = 0
        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()
        self.device = torch.device('cpu')
    
    def save_model(self, path, model_name):
        '''
        Method used for save the model.
        '''
        date = datetime.today().strftime('%Y-%m-%d')
        torch.save(self.model.state_dict(),
                    os.path.join(path, model_name + "__" + self.chk_name + "__" + date + '.pt'))

    def to(self, device):
        self.device = device
        self.adversarial_loss.to(device)
        self.pixelwise_loss.to(device)

    def _early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def _exec(self, input_b, real_b, enable_discriminator, enable_generator):
        D = self.model.D
        batch_size = len(input_b)
        input_b = input_b.to(self.device)
        real_b = real_b.to(self.device)
        r = self.model.get_resolution()
        real_b = real_b[..., (r-32):((2*r)-32), 0:r]

        size = int((self.model.get_resolution() / 8 - 1))
        lab_real = torch.full((batch_size, 1, size, size), 0.9).to(self.device)
        lab_fake = torch.full((batch_size, 1, size, size), 0.1).to(self.device)

        if (enable_generator):
            with torch.enable_grad():
                prediction = self.model(input_b)
        else:
            with torch.no_grad():
                prediction = self.model(input_b)

        if (enable_discriminator):
            with torch.enable_grad():
                D_real = D(real_b)
                D_fake = D(prediction)
        else:
            with torch.no_grad():
                D_real = D(real_b)
                D_fake = D(prediction)

        lossD_real = self.adversarial_loss(torch.sigmoid(D_real), lab_real)
        lossD_fake = self.adversarial_loss(torch.sigmoid(D_fake), lab_fake)

        lossD = lossD_real + lossD_fake

        lossG_adv = self.adversarial_loss(torch.sigmoid(D_fake), lab_real)
        pixelwise_loss_value = self.pixelwise_loss(prediction, real_b)
        lossG = 0.1 * lossG_adv + pixelwise_loss_value

        return lossD, lossG

    def evaluate(self, validation_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            batch_pbar = tqdm(validation_dataset, desc = "Validation - Batch", leave = False)
            for batch in batch_pbar:

                input_b = batch['inputs']
                real_b = batch['reals']
                
                lossD, lossG = self._exec(input_b, real_b, False, False) 

                valid_loss += lossG.tolist()
                batch_pbar.set_postfix({'validation_loss': lossG.tolist(), 'patience': self.counter})
        avg_valid_loss = valid_loss / len(validation_dataset)
        return avg_valid_loss
    
    def _train(self, input_b, real_b, epoch, offset, mode):

        enable_d = epoch < offset or (epoch % 2 != offset % 2) or mode == 'union'
        if enable_d:
            self.model.train_discriminator()
            self.d_optimizer.zero_grad()

        enable_g = epoch >= offset and ((epoch % 2 == offset % 2) or mode == 'union')
        if enable_g:
            self.model.train_generator()
            self.g_optimizer.zero_grad()

        lossD, lossG = self._exec(input_b, real_b, enable_d, enable_g)

        if enable_d:
            lossD.backward()
            self.d_optimizer.step()
        if enable_g:
            lossG.backward()
            self.g_optimizer.step()
        return lossD, lossG

    def train(self, batch_size, train_dataset, validation_dataset, offset = 1, mode ="swap", epochs=100):
        # ciclo sulle epoche per ogni batch
        valid_loss = 1000.0
        self.counter = 0

        # List of training losses
        train_d_losses = []
        train_g_losses = []

        # List of training losses
        valid_losses = []

        for epoch in tqdm(range(epochs), desc = "Epochs", leave = False):
            epoch_d_loss = 0
            epoch_g_loss = 0

            batch_pbar = tqdm(train_dataset, desc = "Training - Batch", leave = True)
            for batch in batch_pbar:
                input_b = batch['inputs']
                real_b = batch['reals']
                self.model.eval_generator()
                self.model.eval_discriminator()

                lossD, lossG = self._train(input_b, real_b, epoch, offset, mode)

                epoch_d_loss += lossD.tolist()
                epoch_g_loss += lossG.tolist() 

                batch_pbar.set_postfix({'v': self.min_validation_loss, 'd': lossD.item(), 'g': lossG.item(), 'p': self.counter})

                avg_epoch_d_loss = epoch_d_loss / len(train_dataset)
                train_d_losses.append(avg_epoch_d_loss)

                avg_epoch_g_loss = epoch_g_loss / len(train_dataset)
                train_g_losses.append(avg_epoch_g_loss)

            print('e_{}: D(x)={:.4f} D(G(z))={:.4f}'.format(epoch, avg_epoch_d_loss, avg_epoch_g_loss))
            if epoch >= offset + 2 and ((epoch % 2 == offset % 2) or mode == 'union'):
                # validation loss
                valid_loss = self.evaluate(validation_dataset)

                # print('val_loss', valid_loss)
                valid_losses.append(valid_loss)
                print('V(x):{}'.format(valid_loss))
                if self._early_stop(valid_loss):
                    #self.save_model('GAN')
                    break

        return train_d_losses, train_g_losses, valid_losses

    def test(self, test_dataset, batch_size):
        self.model.eval()
        metrics = {}
        binary_losses = 0
        MSE_losses = 0
        with torch.no_grad():
            batch_pbar = tqdm(test_dataset, desc = "Test - Batch", leave = False)
            for batch in batch_pbar:
                input = batch['input']
                real = batch['real']

                D = self.model.D
                
                size = int(self.model.get_resolution() / 8 - 1)
                lab_real = torch.full((batch_size, 1, size, size), 0.99).to(self.device)
                
                prediction = self.model(input)

                D_fake = D(prediction)

                lossG_adv = self.adversarial_loss(torch.sigmoid(D_fake),  lab_real)
                pixelwise_loss_value = self.pixelwise_loss(prediction, real)
                lossG = 0.1 * lossG_adv + pixelwise_loss_value
                
                binary_losses += lossG.item()

                MSE_loss = F.mse_loss(prediction, real)
                MSE_losses += MSE_loss.item()

        metrics['cross_entropy'] = binary_losses / len(test_dataset)
        metrics['MSE_entropy'] = MSE_losses / len(test_dataset)
        return metrics
