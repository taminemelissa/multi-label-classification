from src.models.NN.base_model import *


class MixNet1(BaseNet):
    def __init__(self):
        super(MixNet1, self).__init__()
        self.data_type = 2
        self.fc1 = nn.Linear(384, 21)  # linear layer (384 -> 21)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MixNet2(BaseNet):  
    def __init__(self):
        super(MixNet2,self).__init__()
        self.data_type = 2
        self.fc1 = nn.Linear(384, 256)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,21)
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        outpout = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output