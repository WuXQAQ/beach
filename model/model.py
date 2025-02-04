import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size,output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # 第一层
            nn.LeakyReLU(),
            nn.Linear(128, 64),  # 第一层
            nn.LeakyReLU(),
            nn.Linear(64, 32),  # 第一层
            nn.LeakyReLU(),
            nn.Linear(32, 32),  # 第一层
            nn.LeakyReLU(),
            nn.Linear(32, output_size)

        )#无海堤
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, 512),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 128),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 64),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 32),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(32, output_size)
        #
        # )
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, 64),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 32),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 16),  # 第一层
        #     nn.LeakyReLU(),
        #     nn.Linear(16, output_size)
        # )

    def forward(self, x):
        return self.network(x)

# 总网络结构
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.mlp1 = MLP(input_size=17, output_size=2)  # 根据特征数量调整input_size

    def forward(self, features, x):
        A_B = self.mlp1(features)  # 输出A和B
        A, B = A_B[:, 0], A_B[:, 1]
        B = torch.maximum(B, torch.tensor(1e-10))
        output=A*(x**B)
        # output = torch.log(A)+B*torch.log(x)  # H- Ax^B
        return output


