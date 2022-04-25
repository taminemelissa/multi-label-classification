from src.models.NN.base_model import *


class AudioNet1(BaseNet):
    def __init__(self):
        super(AudioNet1, self).__init__()
        self.data_type = 0
        self.fc1 = nn.Linear(256, 21)  # linear layer (256 -> 22)

    def forward(self, x):
        x = self.fc1(x)
        return x
