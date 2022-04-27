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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(126, 21)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.fc1(out)
        out = out.reshape([1,21])
        return out