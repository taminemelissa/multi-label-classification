import torch
import torch.nn as nn
from tqdm import tqdm


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.name = type(self).__name__
        self.data_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train(self, model, dataloader, optimizer, criterion, train_data):
        print('Training')
        model.train()
        device = self.device
        counter = 0
        train_running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
            counter += 1
            if self.data_type == 0:
                data, target = data['audio_feature'].to(device), data['label'].to(device)
            elif self.data_type == 1:
                data, target = data['usage_feature'].to(device), data['label'].to(device)
            else:
                data, target = data['feature'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            train_running_loss += loss.item()
            # backpropagation
            loss.backward()
            # update optimizer parameters
            optimizer.step()

        train_loss = train_running_loss / counter
        return train_loss

    def _validate(self, model, dataloader, criterion, val_data):
        print('Validating')
        model.eval()
        device = self.device
        counter = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
                counter += 1
                if self.data_type == 0:
                    data, target = data['audio_feature'].to(device), data['label'].to(device)
                elif self.data_type == 1:
                    data, target = data['usage_feature'].to(device), data['label'].to(device)
                else:
                    data, target = data['feature'].to(device), data['label'].to(device)
                outputs = model(data)
                # apply sigmoid activation to get all the outputs between 0 and 1
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, target)
                val_running_loss += loss.item()

            val_loss = val_running_loss / counter
            return val_loss
