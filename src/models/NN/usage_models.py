from src.models.NN.base_model import *


class UsageNet1(BaseNet):
    def __init__(self):
        super(UsageNet1, self).__init__()
        self.data_type = 1
        self.fc1 = nn.Linear(128, 21)  # linear layer (128 -> 22)

    def forward(self, x):
        x = self.fc1(x)
        return x
