import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from dataset import StockDataset
from model import TimeSeriesTransformer

import logging

# Set up logging configuration
logging.basicConfig(
    filename='pipeline.log',              # Log file path
    level=logging.INFO,              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set MLflow experiment
mlflow.set_experiment("Stock_TimeSeries_Transformer")

# Load data
X = np.load('data/balanced/X.npy')
y = np.load('data/balanced/y.npy')

logging.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

# Dataset
dataset = StockDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TimeSeriesTransformer().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training parameters
epochs = 10
best_val_acc = 0.0

if __name__ == '__main__':
# Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 🛠️ Fix the shape issue: ensure X_batch is 3D
                if X_batch.dim() == 2:
                    X_batch = X_batch.unsqueeze(1)  # (batch_size, 1, input_dim)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Log training loss
            train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    # 🛠️ Again fix the shape
                    if X_batch.dim() == 2:
                        X_batch = X_batch.unsqueeze(1)  # (batch_size, 1, input_dim)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            # Log validation metrics
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            # Save model if it has the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save model inside the run
                mlflow.pytorch.log_model(model, "model")
                mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Log best validation accuracy at the end
        mlflow.log_metric("final_best_val_accuracy", best_val_acc)
