import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.ard_model import CombinedModelwithARD
from util.Dataset import ExcelDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 32
alpha = 2.0
num_samples = 100
criterion=nn.MSELoss()

# 初始化模型集合
models = []
for fold in range(1,2):
    model = CombinedModelwithARD(torch.tensor([2.0] * 19)).to(device)
    model.load_state_dict(torch.load(f"./saved_models/1/fold_{fold}_best.pth")["model_state"])
    models.append(model)

def predict(_models,_dataloader):
    for _model in _models:
        _model.train()
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for features, x, y, H in tqdm(_dataloader, desc="MC Testing"):
                features, x, y, H = features.to(device), x.to(device), y.to(device), H.to(device)
                target = H - y
                # 蒙特卡洛采样
                mc_preds = []
                for _ in range(num_samples):
                    pred = _model(features, x)
                    mc_preds.append(pred)
                mc_preds = torch.stack(mc_preds)  # [num_samples, B, ...]
                all_preds.append(mc_preds)
                all_targets.append(target)

                # 合并结果
            all_preds = torch.cat(all_preds, dim=1)  # [num_samples, total_samples, ...]
            all_targets = torch.cat(all_targets, dim=0)  # [total_samples, ...]

            # 计算统计量
            mean_pred = (all_preds.mean(dim=0)).mean(dim=1)
            std_pred = (all_preds.std(dim=0)).mean(dim=1)
            all_targets = all_targets.view(-1)
            x=x.view(-1)
            plot_predictions_with_uncertainty(x,all_targets,mean_pred,std_pred)
    return

def plot_predictions_with_uncertainty(
        x: torch.Tensor,
        all_targets: torch.Tensor,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
        title: str = "Predictions vs True Values",
        xlabel: str = "X Value",
        ylabel: str = "Target Value",
        save_path: str = None,
        show_plot: bool = True
) -> plt.Figure:
    """
    可视化预测结果与不确定性

    参数:
    x -- 输入特征的一维张量 [N]
    all_targets -- 真实值的一维张量 [N]
    mean_pred -- 预测均值的一维张量 [N]
    std_pred -- 预测标准差的一维张量 [N]
    title -- 图表标题 (默认: "Predictions vs True Values")
    xlabel -- X轴标签 (默认: "X Value")
    ylabel -- Y轴标签 (默认: "Target Value")
    save_path -- 图片保存路径 (默认: None)
    show_plot -- 是否显示图表 (默认: True)

    返回:
    matplotlib的Figure对象
    """
    # 维度验证
    assert x.dim() == 1, "x应为1维张量"
    assert all_targets.dim() == 1, "all_targets应为1维张量"
    assert mean_pred.dim() == 1, "mean_pred应为1维张量"
    assert std_pred.dim() == 1, "std_pred应为1维张量"
    assert len(x) == len(all_targets) == len(mean_pred) == len(std_pred), "所有输入张量长度必须一致"

    # 转换为CPU上的NumPy数组
    x_np = x.cpu().numpy()
    y_true_np = all_targets.cpu().numpy()
    y_pred_np = mean_pred.cpu().numpy()
    y_std_np = std_pred.cpu().numpy()

    # 按x值排序
    sorted_idx = np.argsort(x_np)
    x_sorted = x_np[sorted_idx]
    y_true_sorted = y_true_np[sorted_idx]
    y_pred_sorted = y_pred_np[sorted_idx]
    y_std_sorted = y_std_np[sorted_idx]

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 3))

    # 绘制真实值散点
    ax.scatter(
        x_sorted, y_true_sorted,
        c='#1f77b4',  # 蓝色
        s=40,  # 点大小
        alpha=0.7,  # 透明度
        edgecolors='w',  # 白色边缘
        label='True Values',
        zorder=3  # 显示在最上层
    )

    # 绘制预测均值曲线
    ax.plot(
        x_sorted, y_pred_sorted,
        color='#d62728',  # 红色
        linewidth=2.5,  # 线宽
        label='Predicted Mean',
        zorder=2
    )

    # 填充不确定性区域
    ax.fill_between(
        x_sorted,
        y_pred_sorted - y_std_sorted,
        y_pred_sorted + y_std_sorted,
        color='#ff7f0e',  # 橙色
        alpha=0.3,  # 透明度
        label='Uncertainty (±1σ)',
        zorder=1
    )

    # 图表装饰
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.8,
        fontsize=10
    )

    # 自动调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 控制显示
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

def test(folder="./Dataset/test/"):
    # 处理测试文件
    test_folder = folder
    for file in os.listdir(test_folder):
        if file.endswith('.xlsx'):
            file_path = os.path.join(test_folder, file)
            print(f"Processing {file}...")
            test_dataset = ExcelDataset(file_path)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            # 执行预测
            predict(models, test_loader)
test("./Dataset/test")

