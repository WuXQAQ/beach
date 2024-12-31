import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于进度条

from model.model import CombinedModel
from util.Dataset import ExcelDataset

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
train_file_path = "D:\BaiduSyncdisk\Beach/beach\Dataset\无海堤/train.xlsx"  # 训练集Excel文件路径
test_file_path = "D:\BaiduSyncdisk\Beach/beach\Dataset\无海堤/val.xlsx"  # 测试集Excel文件路径

train_dataset = ExcelDataset(train_file_path)
test_dataset = ExcelDataset(test_file_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 洗牌训练数据
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型、损失函数和优化器
model = CombinedModel().to(device)  # 将模型移动到选定的设备上
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器 (可选)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
torch.autograd.set_detect_anomaly(True)
# 训练和验证
epochs = 10
best_val_loss = float('inf')  # 用于早停机制
patience = 5  # 早停机制的耐心值
patience_counter = 0  # 当前未见改进的轮数计数器

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    print(f"Epoch {epoch + 1}/{epochs}: Training")
    train_progress = tqdm(train_loader, desc="Training", leave=False)  # 训练进度条
    for features1, features2, x, y, H in train_progress:
        features1, features2, x, y ,H = features1.to(device), features2.to(device), x.to(device), y.to(device), H.to(device)  # 移动数据到选定的设备上
        optimizer.zero_grad()
        predictions = model(features1, features2, x,H)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y) / len(train_loader.dataset)  # 平均批次损失
        train_progress.set_postfix(loss=loss.item())  # 在进度条上显示当前批次的损失

    # 验证
    model.eval()
    val_loss = 0
    total_samples = 0
    print(f"Epoch {epoch + 1}/{epochs}: Validation")
    val_progress = tqdm(test_loader, desc="Validation", leave=False)  # 验证进度条
    with torch.no_grad():
        for features1, features2, x, y,H in val_progress:
            features1, features2, x, y,H = features1.to(device), features2.to(device), x.to(device), y.to(device),H.to(device)  # 移动数据到选定的设备上
            predictions = model(features1, features2, x,H)
            loss = criterion(predictions, y)  # 当前批次的MSE
            val_loss += loss.item() * len(y)  # 累加总误差（乘以样本数）
            total_samples += len(y)  # 统计样本总数
            val_progress.set_postfix(loss=loss.item())  # 在进度条上显示当前批次的损失

    val_mse = val_loss / total_samples  # 计算验证集的平均MSE

    # 更新学习率
    scheduler.step(val_mse)

    # 输出训练损失和验证MSE
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation MSE: {val_mse:.4f}")

    # 早停机制
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break



