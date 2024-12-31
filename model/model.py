import torch
import torch.nn as nn



# 定义MLP模块
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

# 总网络结构
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.mlp1 = MLP(input_size=17, hidden_size=32, output_size=1)  # 根据特征数量调整input_size

        self.mlp2 = MLP(input_size=17, hidden_size=32, output_size=1)

    def forward(self, features1, features2, x,H):
        A = self.mlp1(features1)  # 输出A
        B = self.mlp2(features2)  # 输出B
        output = H - A * (x ** B)  # H- Ax^B
        return output


