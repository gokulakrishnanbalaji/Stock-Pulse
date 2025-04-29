from fetch_data import fetch_data
from preprocess import process_all_stocks
from balance_data import balance_data

from train import val_loader

import os, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch

# Define paths
raw_data_path = 'data/raw/'
new_raw_data_path = 'data/raw/new'
processed_data_path = 'data/processed/'
model_path = 'models/model/data/model.pth'

if os.path.exists(new_raw_data_path):
    shutil.rmtree(new_raw_data_path)

os.makedirs(new_raw_data_path, exist_ok=True)

if os.path.exists(processed_data_path):
    shutil.rmtree(processed_data_path)

os.makedirs(processed_data_path, exist_ok=True)



# fetch new this week's data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
fetch_data(
    save_path=raw_data_path,
    period='7d',
    interval='1d',
    start_date=start_date.strftime('%Y-%m-%d')
)

fetch_data(
    save_path=new_raw_data_path,
    period='7d',
    interval='1d',
    start_date=start_date.strftime('%Y-%m-%d')
)

# preprocess data
process_all_stocks(
    raw_data_path=new_raw_data_path,
    save_path=processed_data_path
)

# balance data
X = np.load(os.path.join(processed_data_path, 'X.npy'))
y = np.load(os.path.join(processed_data_path, 'y.npy'))

# fetch model from models/model/data/model.pth
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])

model_path = os.path.join(model_path)
model = torch.load(model_path, weights_only=False)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 10
best_val_acc = 0.0

mlflow.set_experiment("Stock_TimeSeries_Transformer Finetuning v2")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "BCELoss")

    # Finetuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if batch_X.dim() == 2:
                batch_X = batch_X.unsqueeze(1)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)

        print(f'Epoch {epoch} : Train loss - {avg_loss}')
        
        
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        
        model.train()

# Step 6: Update the finetuned model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model, model_path)
print(f"Updated finetuned model saved to {model_path}")