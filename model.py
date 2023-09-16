import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class My_Model(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.liner_1 = nn.Linear(28 * 28, 120)  # 输入特征维度28*28
    #     self.liner_2 = nn.Linear(120, 84)
    #     self.liner_3 = nn.Linear(84, 10)  # 十分类故输出应为十
    #
    # def forward(self, input):
    #     x = input.view(-1, 28 * 28)
    #     x = F.relu(self.liner_1(x))
    #     x = F.relu(self.liner_2(x))
    #     x = self.liner_3(x)
    #     return x

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
        # ).to("cuda:0")

    def forward(self, input):
        input = np.squeeze(input)
        input = input.view(-1, 28*28)
        # input = torch.tensor(input, device="cuda:0")
        output = self.model(input)
        return output
