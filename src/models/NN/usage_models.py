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

class UsageNet3(BaseNet):  

    def __init__(self):
        super(UsageNet3,self).__init__()
        self.data_type = 1
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 104)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(104,52)
        self.fc3 = nn.Linear(52,21)
        self.norm1 = nn.BatchNorm1d(104)
        self.norm2 = nn.BatchNorm1d(52)
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