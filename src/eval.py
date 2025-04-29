import torch
import numpy as np
import mlflow.pytorch
from dataset import StockDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)

from train import val_loader

import logging

# Set up logging configuration
logging.basicConfig(
    filename='pipeline.log',              # Log file path
    level=logging.INFO,              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)



# Set experiment and tracking uri (optional if default)
mlflow.set_experiment("Stock_TimeSeries_Transformer")

# 1. Find the best run (latest run for now)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Stock_TimeSeries_Transformer")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.final_best_val_accuracy DESC"],  # Sort by best validation accuracy
    max_results=1
)
best_run_id = runs[0].info.run_id
logging.info(f"✅ Best Run ID: {best_run_id}")

# 2. Download and load the model
import os
import mlflow.pytorch
import torch

# Find best model from MLflow
model_uri = f"runs:/{best_run_id}/model"  # Model name must match your log_model() call

# Create a local 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download and load the model
local_model_path = "models"
mlflow.pytorch.load_model(model_uri, dst_path=local_model_path)

# Allowlist the Linear module to safely load the model
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])

# Now load it from the local path
model_path = os.path.join(local_model_path, "model", "data", "model.pth")
model = torch.load(model_path, weights_only=False)  # Explicitly set weights_only=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)

        if X_batch.dim() == 2:
            X_batch = X_batch.unsqueeze(1) 
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())


# Metrics
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

logging.info(f"🎯 Accuracy: {acc:.4f}")
logging.info(f"🎯 F1 Score: {f1:.4f}")
logging.info(f"🎯 Precision: {precision:.4f}")
logging.info(f"🎯 Recall: {recall:.4f}")

# Confusion Matrix Plot
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix')

# Save the plot
os.makedirs("outputs", exist_ok=True)
conf_matrix_path = "outputs/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.close()
logging.info(f"✅ Confusion matrix saved at {conf_matrix_path}")

# ---- Log everything to MLflow ----
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_f1_score", f1)
    mlflow.log_metric("eval_precision", precision)
    mlflow.log_metric("eval_recall", recall)
    mlflow.log_artifact(conf_matrix_path)

logging.info("✅ Metrics and confusion matrix logged to MLflow")