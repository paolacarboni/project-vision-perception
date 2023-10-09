from datetime import datetime
import torch
import os
from tqdm.auto import tqdm
import torch.nn.functional as F

class UNetTrainer():
    def __init__(self, model, optimizer, loss, chk_name, patience=1, min_delta=0):
        self.model = model
        self.optimizer = optimizer
        self.chk_name = chk_name
        self.iteration = 0
        self.patience = patience
        self.min_validation_loss = float('inf')
        self.counter = 0
        self.min_delta = min_delta
        self.loss = loss

    def save_model(self, path, model_name):
        '''
        Method used for save the model.
        '''
        date = datetime.today().strftime('%Y-%m-%d')
        torch.save(self.model.state_dict(),
                    os.path.join(path, model_name + "__" + self.chk_name + "__" + date + '.pth'))

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


    def train(self, train_dataset, validation_dataset, epochs=100):
        # ciclo sulle epoche per ogni batch
        valid_loss = 1000.0

        # List of training losses
        train_losses = []

        # List of training losses
        valid_losses = []

        for epoch in tqdm(range(epochs), desc = "Epochs", leave = False):
            epoch_loss = 0
            self.model.train()

            batch_pbar = tqdm(train_dataset, desc = "Training - Batch", leave = False)
            for batch in batch_pbar:
                self.optimizer.zero_grad()

                input = batch['input']
                mask = batch['mask']

                prediction = self.model(input)

                # Compute gradient
                sample_loss = self.loss(prediction, mask)

                sample_loss.backward()

                self.optimizer.step()

                epoch_loss += sample_loss.tolist()
                batch_pbar.set_postfix({'validation_loss': valid_loss, 'training_loss': sample_loss.item(), 'patience': self.counter})

            # training epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_losses.append(avg_epoch_loss)
            # validation loss
            valid_loss = self.evaluate(validation_dataset, self.loss)

            # print('val_loss', valid_loss)
            valid_losses.append(valid_loss)
            if self.early_stop(valid_loss):
                self.save_model('UNET')
                break
        return train_losses, valid_losses

    def evaluate(self, validation_dataset, loss):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            batch_pbar = tqdm(validation_dataset, desc = "Validation - Batch", leave = False)
            for  batch in batch_pbar:

                input = batch['input']
                mask = batch['mask']
                prediction = self.model(input) #pred

                # Compute gradient
                sample_loss = loss(prediction, mask)

                valid_loss += sample_loss.tolist()
                batch_pbar.set_postfix({'validation_loss': sample_loss.tolist(), 'patience': self.counter})
        avg_valid_loss = valid_loss / len(validation_dataset)
        return avg_valid_loss



    def test(self, test_dataset):
        self.model.eval()
        metrics = {}
        binary_losses = 0
        MSE_losses = 0
        with torch.no_grad():
            batch_pbar = tqdm(test_dataset, desc = "Test - Batch", leave = False)
            for batch in batch_pbar:
                input = batch['input']
                mask = batch['mask']
                prediction = self.model(input)

                binary_loss = self.loss(prediction, mask)
                binary_losses += binary_loss.item()

                MSE_loss = F.mse_loss(prediction, mask)
                MSE_losses += MSE_loss.item()
        metrics['cross_entropy'] = binary_losses / len(test_dataset)
        metrics['MSE_entropy'] = MSE_losses / len(test_dataset)
        return metrics
