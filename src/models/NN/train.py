if __name__ == '__main__':
    
    import os
    import sys
    project_dir = os.getcwd().split('src')[0]
    sys.path.append(project_dir)

    import pandas as pd
    import torch.optim as optim
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    from src.models.NN.audio_models import *
    from src.models.NN.usage_models import *
    from src.models.NN.mix_models import *
    from src.models.NN.dataset import TrackDataset
    from torch.utils.data import DataLoader
    from src.utils.tools import load_json
    from src.utils.save_best_model import *

    # load the config file
    config = load_json(os.path.join(project_dir, "src/models/NN/config.json"))

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    model = AudioNet1()  # initialize the neural network
    model.to(device=device)

    # learning parameters
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # read the parquet file
    df = pd.read_parquet(os.path.join(project_dir,"dataset2.parquet"), engine="pyarrow")

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
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = model._train(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, train_data=train_data)
        valid_epoch_loss = model._validate(model=model, dataloader=valid_loader, criterion=criterion, val_data=valid_data)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion)
        print('-' * 50)

    # save the final trained model to disk
    #torch.save({
        #'epoch': epochs,
        #'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        #'loss': criterion,}, os.path.join(project_dir, 'final_model.pth'))

    # plot and save the train and validation graphs
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(project_dir,f"docs/outputs/NN/loss_{model.name}.png"))
    plt.show()
