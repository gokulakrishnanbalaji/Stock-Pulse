import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
