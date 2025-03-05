import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.ard_model import CombinedModelwithARD
from util.Dataset import ExcelDataset
import os

#device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 32
alpha = 2.0
num_samples = 100
criterion=nn.MSELoss()

test_file_path = "./Dataset/test.xlsx"
test_dataset = ExcelDataset(test_file_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

alpha_list = torch.tensor([alpha] * 19)

def mc_test(_model, data_loader, _criterion, _num_samples=num_samples):
    """带MC Dropout的测试函数（输出RMSE）"""
    _model.train()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for features, x, y, H in tqdm(data_loader, desc="MC Testing"):
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
    mean_pred = all_preds.mean(dim=0)
    std_pred = all_preds.std(dim=0)

    # 计算指标（关键修改点）
    mse = _criterion(mean_pred, all_targets)
    rmse = torch.sqrt(mse).item()  # MSE -> RMSE
    uncertainty = std_pred.mean().item()

    return rmse, uncertainty  # 返回RMSE
test_results = []
for fold_idx in range(5):
    print(f"\n=== Testing Fold {fold_idx + 1} ===")

    # 加载对应折的最佳模型
    model = CombinedModelwithARD(alpha_list).to(device)
    checkpoint = torch.load(f'./saved_models/fold_{fold_idx + 1}_best.pth')
    model.load_state_dict(checkpoint['model_state'])

    # 执行测试
    test_rmse, test_uncertainty = mc_test(model, test_loader, criterion)
    test_results.append(test_rmse)

    print(f"Fold {fold_idx + 1} Test RMSE: {test_rmse:.4f} ± {test_uncertainty:.4f}")

# 计算平均性能
final_rmse = sum(test_results) / len(test_results)
print(f"\nFinal Average Test RMSE: {final_rmse:.4f}")