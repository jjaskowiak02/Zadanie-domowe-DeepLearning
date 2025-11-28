import torch
from torch.utils.data import Dataset
import pandas as pd

def load_data(csv_file: str) -> tuple:
    data = pd.read_csv(csv_file)
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).astype(float)
    
    features = pd.DataFrame(data.drop(columns=['Calories', 'id'], errors='ignore'))
    target = pd.DataFrame(data['Calories']) if 'Calories' in data.columns else None
    
    x = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1) if target is not None else None
    
    return x, y

class WorkoutDataset(Dataset):
    def __init__(self, csv_file: str, normalize: bool = True):
        x, y = load_data(csv_file)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        
        if normalize:
            x_min = self.x.min(dim=0, keepdim=True).values
            x_max = self.x.max(dim=0, keepdim=True).values
            self.x = (self.x - x_min) / (x_max - x_min + 1e-7)
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx) -> tuple:
        return self.x[idx], self.y[idx]
