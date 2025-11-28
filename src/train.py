import os
import logging
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss

logger = logging.getLogger(__name__)

def train(dataset, dataloader, model, epochs=80, lr=0.002, momentum=0.85):
    criterion = RMSLELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    best_rmsle = float('inf')
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        running_loss /= len(dataloader)
        losses.append(running_loss)
        
        if epoch % 8 == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.tensor(model(dataset.x))
                rmsle = torch.tensor(criterion(pred, dataset.y))
            logger.info(f"Epoch {epoch+1}/{epochs} - RMSLE: {rmsle:.3f} - Loss: {running_loss:.4f}")
            
            if rmsle.item() < best_rmsle:
                best_rmsle = rmsle.item()
                save_path = str(os.path.join(HydraConfig.get().run.dir, 'best_model.pt'))
                torch.save(model.state_dict(), save_path)
            
            model.train()
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig(os.path.join(HydraConfig.get().run.dir, 'loss_curve.png'))
    plt.close()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = WorkoutDataset(cfg.data.train)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    model = NeuralNetwork()
    train(dataset, dataloader, model, epochs=cfg.training.epochs, lr=cfg.training.learning_rate, momentum=cfg.training.momentum)

if __name__ == "__main__":
    main()
