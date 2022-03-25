import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from models import *
from dataset import TrackDataset
from torch.utils.data import DataLoader
import os
matplotlib.style.use('ggplot')


#load the config file
config = load_json('config.json')

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = config['model'][0] # initialize the neural network
model.to(device=device)

# learning parameters
lr = config['lr']
epochs = config['epochs']
batch_size = config['batch_size']
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# read the parquet file
df = pd.read_parquet("dataset2.parquet", engine="pyarrow")

# train dataset
train_data = TrackDataset(df, train=True, test=False)

# validation dataset
valid_data = TrackDataset(df, train=False, test=False)

# train data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# validation data loader
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
    valid_epoch_loss = validate(model, valid_loader, criterion, valid_data, device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')
    
# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, os.path.join(config['checkpoints'], 'best-checkpoint.pth'))

# plot and save the train and validation graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../loss.png')
plt.show()
