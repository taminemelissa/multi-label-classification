from src.models.NN.base_model import *


class UsageNet1(BaseNet):
    def __init__(self):
        super(UsageNet1, self).__init__()
        self.data_type = 1
        self.fc1 = nn.Linear(128, 21)  # linear layer (128 -> 21)

    def forward(self, x):
        x = self.fc1(x)
        return x

class UsageNet2(BaseNet):  
    def __init__(self):
        super(UsageNet2,self).__init__()
        self.data_type = 1
        self.fc1 = nn.Linear(128, 104)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(104,104)
        self.fc3 = nn.Linear(104,21)
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        outpout = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output