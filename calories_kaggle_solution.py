#
#Uruchomienie treningu:
#        python calories_kaggle_solution.py --mode train --train train.csv --test test.csv --out outputs
#-''- predykcji:
#        python calories_kaggle_solution.py --mode predict --model outputs/best_model.pth --train train.csv --test test.csv --out_submission outputs/submission.csv


import os
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CaloriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32).reshape(-1,1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# model
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32,16], dropout=0.0, use_activation=True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_activation:
                layers.append(nn.ReLU())
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

#przygotowanie danych
def prepare_data(train_csv=None, test_csv=None, seed=42):
    train_df = pd.read_csv(train_csv)
    train_df.columns = [c.strip() for c in train_df.columns]
    if "Calories" not in train_df.columns:
        raise KeyError("Kolumna 'Calories' nie istnieje w train.csv")
    test_df = pd.read_csv(test_csv)
    test_df.columns = [c.strip() for c in test_df.columns]
    cat_cols = ["Sex"]
    combined = pd.concat([train_df.drop(columns=["Calories"]), test_df], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    X_train = combined.iloc[:len(train_df)].values.astype(np.float32)
    y_train = train_df["Calories"].values.astype(np.float32)
    X_test = combined.iloc[len(train_df):].values.astype(np.float32)
    return X_train, y_train, test_df, combined.columns.tolist()

# trening
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses=[]
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

def eval_model(model, dataloader, criterion, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds, trues = np.vstack(preds), np.vstack(trues)
    return mean_squared_error(trues, preds)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# start
def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, test_df, feature_cols = prepare_data(args.train, args.test)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    ensure_dir(args.out)
    results=[]

    for lr in args.lr_list:
        for batch in args.batch_list:
            for dropout in args.dropout_list:
                for hidden in args.hidden_configs:
                    hidden_dims = [int(v) for v in hidden.split('-')]
                    model = SimpleRegressor(X.shape[1], hidden_dims=hidden_dims, dropout=dropout, use_activation=not args.no_activation).to(device)
                    train_loader = DataLoader(CaloriesDataset(X_train, y_train), batch_size=batch, shuffle=True)
                    val_loader = DataLoader(CaloriesDataset(X_val, y_val), batch_size=256, shuffle=False)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()
                    best_val = float('inf')
                    for epoch in range(args.epochs):
                        train_one_epoch(model, train_loader, optimizer, criterion, device)
                        val_mse = eval_model(model, val_loader, criterion, device)
                        if val_mse < best_val:
                            best_val = val_mse
                            torch.save({'state_dict': model.state_dict(),
                                        'config': {'hidden_dims': hidden_dims, 'dropout': dropout, 'use_activation': not args.no_activation}},
                                       os.path.join(args.out, 'temp_best.pth'))
                    results.append({'lr':lr,'batch':batch,'dropout':dropout,'hidden':hidden_dims,'val_mse':best_val})
                    print(f"lr={lr} batch={batch} dropout={dropout} hidden={hidden_dims} val_mse={best_val:.4f}")

    res_df = pd.DataFrame(results).sort_values('val_mse')
    res_df.to_csv(os.path.join(args.out,'grid_results.csv'), index=False)
    best_row = res_df.iloc[0]
    best_ckpt = torch.load(os.path.join(args.out,'temp_best.pth'))
    model = SimpleRegressor(X.shape[1], hidden_dims=best_ckpt['config']['hidden_dims'],
                            dropout=best_ckpt['config']['dropout'],
                            use_activation=best_ckpt['config']['use_activation'])
    model.load_state_dict(best_ckpt['state_dict'])
    torch.save({'state_dict': model.state_dict(), 'config': best_ckpt['config']}, os.path.join(args.out,'best_model.pth'))
    print("Najlepszy model zapisany w outputs")

# predict
def predict_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, test_df, feature_cols = prepare_data(args.train, args.test)
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and 'config' in ckpt:
        cfg = ckpt['config']
        model = SimpleRegressor(input_dim=X_train.shape[1],
                                hidden_dims=cfg['hidden_dims'],
                                dropout=cfg['dropout'],
                                use_activation=cfg['use_activation']).to(device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise RuntimeError("checkpoint nie zawiera pełnej konfiguracji modelu")
    model.eval()
    test_X = test_df[feature_cols].values.astype(np.float32)
    dl = DataLoader(CaloriesDataset(test_X), batch_size=256, shuffle=False)
    preds=[]
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.vstack(preds).flatten()
    sub = pd.DataFrame({'id': test_df['id'], 'calories': preds})
    sub.to_csv(args.out_submission, index=False)
    print("zapisano:", args.out_submission)

# cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','predict'], required=True)
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--out_submission', type=str, default='outputs/submission.csv')
    parser.add_argument('--model', type=str, default='outputs/best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr_list', nargs='+', type=float, default=[1e-3, 5e-4])
    parser.add_argument('--batch_list', nargs='+', type=int, default=[64,128])
    parser.add_argument('--dropout_list', nargs='+', type=float, default=[0.0,0.2])
    parser.add_argument('--hidden_configs', nargs='+', type=str, default=['32-16','64-32'])
    parser.add_argument('--no_activation', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == 'train':
        run_experiment(args)
    else:
        predict_and_save(args)

if __name__=='__main__':
    main()
