import torch
import torch.nn as nn
from tqdm import tqdm


class BaseNet(nn.Module):
    def __init__(self):
        self.name = type(self).__name__
        self.data_type = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, dataloader, optimizer, criterion, train_data, device):
        print('Training')
        model.train()
        counter = 0
        train_running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
            counter += 1
            if data_type==0:
                data, target = data['audio_feature'].to(device), data['label'].to(device)
            elif data_type==1:
                data, target = data['usage_feature'].to(device), data['label'].to(device)
            else:
                data, target = data['usage_feature'].to(device), data['label'].to(device)
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
    
    def validate(model, dataloader, criterion, val_data, device):
        raise NotImplementedError
        
# training function
def train(model, dataloader, optimizer, criterion, train_data, device, audio_bool=1):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1
        if audio_bool:
            data, target = data['audio_feature'].to(device), data['label'].to(device)
        else:
            data, target = data['usage_feature'].to(device), data['label'].to(device)
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


# validation function
def validate(model, dataloader, criterion, val_data, device, audio_bool=1):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            if audio_bool:
                data, target = data['audio_feature'].to(device), data['label'].to(device)
            else:
                data, target = data['usage_feature'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

        val_loss = val_running_loss / counter
        return val_loss
