if __name__ == '__main__':
    
    import os
    import sys
    project_dir = os.getcwd().split('src')[0]
    sys.path.append(project_dir)
  
    import torch
    import models
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from dataset import TrackDataset
    from torch.utils.data import DataLoader
    from src.models.NN.audio_models import *
    from src.models.NN.usage_models import *
    from src.models.NN.mix_models import *
    
    # load the config file
    config = load_json(os.path.join(project_dir, "src/models/NN/config.json"))
    
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #intialize the model
    model = AudioNet1()  # initialize the neural network
    model.to(device=device)
    
    # load the model checkpoint
    checkpoint = torch.load(config['model_path'])
    
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # read the parquet file
    df = pd.read_parquet(os.path.join(project_dir,"dataset2.parquet"), engine="pyarrow")
    genres = list(df.columns[:21])
    
    # prepare the test dataset and dataloader
    test_data = TrackDataset(df, train=False, test=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    #start the inference loop
    results = dict('actual': [], 'predicted': [])
    for counter, data in enumerate(test_loader):
        if model.data_type == 0:
            data, target = data['audio_feature'].to(device), data['label']
        elif model.data_type == 1:
            data, target = data['usage_feature'].to(device), data['label']
        else:
            data, target = data['feature'].to(device), data['label']
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
        # get the predictions by passing the data through the model
        outputs = model(data)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()
        sorted_outputs = np.sort(outputs[0])
        sorted_indices = np.argsort(outputs[0])
        if sorted_outputs[-3] > 0.3:
            best = sorted_indices[-3:]
        elif sorted_outputs[-2] > 0.4:
            best = sorted_indices[-2:]
        else:
            best = sorted_indices[-1:]
        predictions = []
        values = []
        for i in range(len(best)):
            predictions.append(f"{genres[best[i]]}")
        for i in range(len(target_indices)):
            values.append(f"{genres[best[i]]}")
        results['actual'].append(values)
        results['predicted'].append(predictions)
            