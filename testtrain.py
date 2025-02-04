import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.ard_model import CombinedModelwithARD
# 假设这些模块已经存在并且正确导入
from util.Dataset import ExcelDataset

# 设备选择
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ep = 1e-10
lambda_reg = 1
# 数据加载
train_file_path = "/Users/wuxi/Desktop/workspace/Beach/beach/Dataset/无海堤/standard_train.xlsx"  # 训练集Excel文件路径
test_file_path = "/Users/wuxi/Desktop/workspace/Beach/beach/Dataset/无海堤/standard_val.xlsx"  # 测试集Excel文件路径

train_dataset = ExcelDataset(train_file_path)
test_dataset = ExcelDataset(test_file_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 洗牌训练数据
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
alpha_list = torch.tensor([2.0] * 17) # 为每个输入特征设置相同的 alpha_i


# 模型、损失函数和优化器
model = CombinedModelwithARD(alpha_list).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200)

# 训练和验证
epochs = 200
best_val_loss = float('inf')
patience = 20  # 耐心值
patience_counter = 0


for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    print(f"Epoch {epoch + 1}/{epochs}: Training")
    train_progress = tqdm(train_loader, desc="Training", leave=False)
    for features, x, y, H in train_progress:
        features, x, y, H = features.to(device), x.to(device), y.to(device), H.to(device)
        optimizer.zero_grad()
        predictions = model(features, x)
        loss = criterion(predictions,H-y)
        reg_loss = model.reg_loss()
        # 总损失 = 数据误差 + 正则化项
        total_loss = loss + reg_loss*lambda_reg
        total_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 计算平均损失
        train_loss += loss.item() * len(y) / len(train_loader.dataset)  # 平均批次损失
        train_progress.set_postfix(loss=loss.item())  # 在进度条上显示当前批次的损失
    model.mlp1.print_input_alpha()

    # 验证
    model.eval()
    val_loss = 0
    total_samples = 0
    print(f"Epoch {epoch + 1}/{epochs}: Validation")
    val_progress = tqdm(test_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, x, y, H in val_progress:
            features, x, y, H = features.to(device), x.to(device), y.to(device), H.to(device)
            predictions = model(features, x)
            target = H - y
            loss = criterion(predictions, target)
            val_loss += loss.item() * len(y)
            total_samples += len(y)
            val_progress.set_postfix(loss=loss.item())

    val_mse = val_loss / total_samples

    scheduler.step(val_mse)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation MSE: {val_mse:.4f}")