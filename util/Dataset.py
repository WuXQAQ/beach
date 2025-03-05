import torch
import pandas as pd
from torch.utils.data import Dataset
class ExcelDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_excel(file_path)
        self.features = torch.tensor(data[["Azimuth","Length","averageParticleSize", "averageParticleSize2","sortingFactor","Skewness","peakState","meanAnnualHighTidalRange","meanAnnualTidalRange","waveDirection","frequency","averageWaveHeight","Periodicity","RTR","Omega","HightideSedimentsettlingvelocity","Hb","Hd","averageAnnualPeriodicity"]].values, dtype=torch.float32)  # 替换为实际列名
        self.x = torch.tensor(data["x"].values, dtype=torch.float32).unsqueeze(1)  # x列
        self.y = torch.tensor(data["y"].values, dtype=torch.float32).unsqueeze(1)  # 标签y列
        self.H = torch.tensor(data["H"].values, dtype=torch.float32).unsqueeze(1)  #H列

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.features[idx], self.x[idx], self.y[idx],self.H[idx]