import torch
import pandas as pd
import os
from .model import NeuralNetwork

def predict(test_csv: str, model_path: str, output_csv: str):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    data = pd.read_csv(test_csv)
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).astype(float)
    
    features = data.drop(columns=['id'])
    x = torch.tensor(features.values, dtype=torch.float32)
    
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    x = (x - x_min) / (x_max - x_min + 1e-7)
    
    with torch.no_grad():
        predictions = model(x).squeeze().numpy()
    
    submission = pd.DataFrame({
        'id': data['id'],
        'Calories': predictions
    })
    
    submission.to_csv(output_csv, index=False)
    print(f"Predykcje zapisane do: {output_csv}")

if __name__ == "__main__":
    predict(
        test_csv="data/test.csv",
        model_path="outputs/2025-11-28/11-53-54/best_model.pt",
        output_csv="data/sample_submission.csv"
    )
