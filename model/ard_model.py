
import torch
import torch.nn as nn


class GaussianPriorLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha_i):
        super(GaussianPriorLayer, self).__init__()
        # 初始化权重均值为0，方差由 alpha_i 控制
        self.weights_mean = nn.Parameter(torch.zeros(in_features, out_features))  # 权重的均值
        self.bias = nn.Parameter(torch.zeros(out_features))  # 偏置

        # 设置 alpha_i 用于控制每个权重的方差
        self.alpha_i = nn.Parameter(alpha_i)  # 方差的控制超参数

    def forward(self, x):
        # 使用 alpha_i 来控制标准差，通过标准差来进行重参数化
        # 扩展 alpha_i 为 (in_features, out_features) 的形状
        weights_std = torch.sqrt(self.alpha_i).unsqueeze(1)  # 扩展到 (in_features, 1)
        weights_std = weights_std.expand_as(self.weights_mean)  # 扩展为 (in_features, out_features)

        # 权重通过均值加上方差的影响（这里采用的是重参数化技巧）
        weights = self.weights_mean + weights_std * torch.randn_like(self.weights_mean)  # 重新参数化的权重

        # 线性变换
        return torch.matmul(x, weights) + self.bias

    def regularization_loss(self):
        # 对每个输入特征计算L2正则化加权高斯先验
        reg_loss = 0
        for i in range(self.weights_mean.shape[0]):  # 遍历所有输入特征
            reg_loss += torch.sum(self.weights_mean[i] ** 2) / (2 * self.alpha_i[i])  # L2正则化加权
        return reg_loss


class MLPwithARD(nn.Module):
    def __init__(self, input_size, output_size, alpha_list):
        super(MLPwithARD, self).__init__()
        self.network = nn.Sequential(
            GaussianPriorLayer(input_size, 64, alpha_list),  # 第一层
            nn.LeakyReLU(),
            nn.Linear(64, 64),  # 第二层
            nn.LeakyReLU(),
            nn.Linear(64, 64),  # 第二层
            nn.LeakyReLU(),
            nn.Linear(64, 32),  # 第二层
            nn.LeakyReLU(),
            nn.Linear(32, 32),  # 第三层
            nn.LeakyReLU(),
            nn.Linear(32, 32),  # 第四层
            nn.LeakyReLU(),
            nn.Linear(32, 16),  # 第四层
            nn.LeakyReLU(),
            nn.Linear(16, output_size)  # 输出层
        )

    def forward(self, x):
        return self.network(x)

    def regularization_loss(self):
        # 总的正则化损失是每一层正则化损失的和
        reg_loss = 0
        for layer in self.network:
            if isinstance(layer, GaussianPriorLayer):
                reg_loss += layer.regularization_loss()
        return reg_loss
    def print_input_alpha(self):
        # 打印输入特征的 alpha_i
        input_layer = self.network[0]  # 第一层是输入层
        print(f"Input feature alpha_i: {input_layer.alpha_i}")
        return input_layer.alpha_i

class CombinedModelwithARD(nn.Module):
    def __init__(self, alpha_list):
        super(CombinedModelwithARD, self).__init__()
        self.mlp1 = MLPwithARD(input_size=17, output_size=2, alpha_list=alpha_list)  # 输入大小调整为18，包括随机特征

    def forward(self, features, x):
        # 向输入特征中添加一个随机特征
        random_feature = torch.randn(features.size(0), 1).to(features.device)  # 随机生成一个特征，大小与 batch size 一致
        features_with_random = torch.cat([features], dim=1)  # 将随机特征拼接到输入特征中

        # 通过 MLP 模型计算输出
        A_B = self.mlp1(features_with_random)  # 输出A和B
        A, B = A_B[:, 0], A_B[:, 1]
        B = torch.maximum(B, torch.tensor(1e-10).to(B.device))
        output = A * (x ** B)

        return output

    def reg_loss(self):
        return self.mlp1.regularization_loss()





