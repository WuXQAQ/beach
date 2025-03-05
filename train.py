import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.ard_model import CombinedModelwithARD
from util.Dataset import ExcelDataset
import os

#device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#hyperparameter
ep = 1e-10
lambda_reg = 1
alpha = 2.0
batch_size = 32
learning_rate = 0.001
epochs = 200
patience = 50

'''
isprint : print alpha
'''
def train(_model, data_loader, _criterion, _optimizer, num_epochs, _train_loss, isprint=False):
        # 训练
    _model.train()

    print(f"Epoch {num_epochs + 1}/{epochs}: Training")
    train_progress = tqdm(data_loader, desc="Training", leave=False)
    for features, x, y, H in train_progress:
        features, x, y, H = features.to(device), x.to(device), y.to(device), H.to(device)
        _optimizer.zero_grad()
        predictions = _model(features, x)
        #loss
        loss = _criterion(predictions, H - y)
        reg_loss = _model.reg_loss()
        total_loss = loss + reg_loss * lambda_reg
        total_loss.backward()

        _optimizer.step()

        _train_loss += loss.item() * len(y) / len(data_loader.dataset)
        train_progress.set_postfix(loss=loss.item())  # print loss

    alpha_output = _model.mlp1.print_input_alpha(isprint)
    return _train_loss

def evaluate(_model, data_loader, _criterion, num_epochs, _val_loss, _total_samples):
    _model.eval()
    print(f"Epoch {num_epochs + 1}/{epochs}: Validation")
    val_progress = tqdm(data_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, x, y, H in val_progress:
            features, x, y, H = features.to(device), x.to(device), y.to(device), H.to(device)
            predictions = _model(features, x)
            target = H - y
            loss = _criterion(predictions, target)
            _val_loss += loss.item() * len(y)
            _total_samples += len(y)
            val_progress.set_postfix(loss=loss.item())
    return _val_loss, _total_samples




'''
5 fold cross validation
'''

# data load
fold_names = ['./Dataset/5_fold_cross_validation/1st','./Dataset/5_fold_cross_validation/2nd','./Dataset/5_fold_cross_validation/3rd','./Dataset/5_fold_cross_validation/4th','./Dataset/5_fold_cross_validation/5th']
alpha_list = torch.tensor([alpha] * 19)
criterion = nn.MSELoss()


all_fold_results = []
best_models = {}  # save the best model

os.makedirs('./saved_models', exist_ok=True)

for fold_idx, fold in enumerate(fold_names):
    print(f"\n=== Processing Fold {fold_idx + 1}/5 ===")

    #initialize the model
    model = CombinedModelwithARD(alpha_list).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_path = os.path.join(fold, 'train.xlsx')
    val_path = os.path.join(fold, 'val.xlsx')
    train_dataset = ExcelDataset(train_path)
    val_dataset = ExcelDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        total_samples = 0
        train_loss = train(model,train_loader,criterion,optimizer,epoch,train_loss)
        val_loss,total_samples = evaluate(model,val_loader,criterion,epoch,val_loss,total_samples)
        val_mse = val_loss / total_samples

        #early stop
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            # 保存当前折的最佳模型状态
            best_model_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_mse,
            }
            # save the model
            torch.save(best_model_state, f'./saved_models/fold_{fold_idx + 1}_best.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation MSE: {val_mse:.4f}")

    best_models[fold_idx] = best_model_state
    all_fold_results.append({
        'fold': fold_idx + 1,
        'best_epoch': best_model_state['epoch'],
        'best_val_loss': best_model_state['val_loss']
    })


