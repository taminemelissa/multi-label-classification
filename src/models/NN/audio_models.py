from src.models.NN.base_model import *


class AudioNet1(BaseNet):
    def __init__(self):
        super(AudioNet1, self).__init__()
        self.data_type = 0
        self.fc1 = nn.Linear(256, 21)  # linear layer (256 -> 21)

    def forward(self, x):
        x = self.fc1(x)
        return x

class AudioNet2(BaseNet):  
    def __init__(self):
        super(AudioNet2,self).__init__()
        self.data_type = 0
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,21)
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        outpout = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output
    
class AudioNet3(BaseNet):  

    def __init__(self):
        super(AudioNet3,self).__init__()
        self.data_type = 0
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,21)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(21)
    
    def forward(self, x):
        output = self.dropout(x)
        output = self.fc1(output)
        output = self.norm1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.norm3(output)
        return output