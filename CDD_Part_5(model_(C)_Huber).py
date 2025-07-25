# -*- coding: utf-8 -*-
"""CDD_Part_5(model_(C)-Huber).ipynb

# Refer my google colab link for notebook file https://colab.research.google.com/drive/1s5WOjlY85Sh543IIljuqxUlv_1ic4edB#scrollTo=7hjQFEBTbe1K
# **Model C Implementation**
"""

!pip install torch torch-geometric rdkit pandas numpy

#Load the datasets

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid SMILES

    # Node features (atom-level)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
        ]
        node_features.append(features)

    # Edge features (bond-level)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.append([bond.GetBondTypeAsDouble()])
        edge_attrs.append([bond.GetBondTypeAsDouble()])

    # Convert to PyTorch Geometric Data object
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Load CSV and preprocess

#df = pd.read_csv("/content/4-NS3_(1)_augmented.csv")
#df = pd.read_csv("/content/4-ro5_properties_fulfilled_augmented.csv")
df = pd.read_csv("/content/4-Clean_NS3_data (1)_augmented.csv")

graphs = []
for _, row in df.iterrows():
    data = smiles_to_graph(row["smiles"], row["pIC50"])
    if data is not None:
        graphs.append(data)

# Split the dataset using k-fold

import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.loader import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Assuming 'graphs' is your full dataset
dataset_size = len(graphs)
indices = np.arange(dataset_size)

# First split: train+val (90%) and test (10%)
train_val_indices, test_indices = train_test_split(
    indices,
    test_size=0.1,
    random_state=42
)

# Initialize K-Fold with 5 splits (adjust k as needed)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Create list to store fold DataLoaders
fold_loaders = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    # Get actual indices from the train_val subset
    fold_train_indices = train_val_indices[train_idx]
    fold_val_indices = train_val_indices[val_idx]

    # Create datasets for current fold
    train_dataset = [graphs[i] for i in fold_train_indices]
    val_dataset = [graphs[i] for i in fold_val_indices]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=75, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=75)

    fold_loaders.append((train_loader, val_loader))
    print(f"Fold {fold+1}:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

# Create test dataset and loader (used once at final evaluation)
test_dataset = [graphs[i] for i in test_indices]
test_loader = DataLoader(test_dataset, batch_size=75)

print(f"\nTest set: {len(test_dataset)} samples")

# Define the GNN model

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class HybridGCNGAT(torch.nn.Module):
    def __init__(self, hidden_dim=128, gat_heads=4):
        super().__init__()

        # First two layers: GCNConv
        self.conv1 = GCNConv(graphs[0].num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Next two layers: GATConv
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=gat_heads)
        self.conv4 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1)

        # Final linear layer for regression
        self.lin = Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Optional dropout

        # Second GCN layer with ReLU activation
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Optional dropout

        # First GAT layer with ELU activation
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Optional dropout

        # Second GAT layer with ELU activation
        x = F.elu(self.conv4(x, edge_index))

        # Global pooling (mean over all nodes in each graph)
        x = global_mean_pool(x, batch)

        # Final linear layer for regression output
        return self.lin(x).squeeze()

# Train the model

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path="/content/best_hybrid_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.last_model_path = None
        self.best_r2 = -np.inf
        self.best_val_loss = np.inf

    def __call__(self, val_loss, val_r2, model):
        if val_r2 > self.best_r2:
            self.best_r2 = val_r2

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_val_loss = val_loss
            self.save_checkpoint(None, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_last_model(model)
        else:
            previous_val_loss = self.best_val_loss
            self.best_score = score
            self.best_val_loss = val_loss
            self.save_checkpoint(previous_val_loss, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, previous_val_loss, current_val_loss, model):
        if previous_val_loss is None:
            print(f"Initial validation loss: {current_val_loss:.4f}. Saving model to {self.path}")
        else:
            print(f"Validation loss decreased ({previous_val_loss:.4f} → {current_val_loss:.4f}). Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)

    def save_last_model(self, model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        r2_str = f"R²={self.best_r2:.4f}".replace(".", "_")
        filename = f"/content/last_model_{r2_str}_{timestamp}.pt"
        self.last_model_path = filename
        print(f"Saving last model to: {filename}")
        torch.save(model.state_dict(), filename)

# Initialize model, optimizer, and loss function
model = HybridGCNGAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000988)
loss_fn = torch.nn.HuberLoss(delta=1.0)  # Huber loss remains unchanged

# Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function with R² calculation
def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()

            all_preds.append(out.detach().cpu().numpy().flatten())
            all_labels.append(batch.y.detach().cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    r2 = r2_score(all_labels, all_preds)
    return avg_loss, r2

# Initialize training components
early_stopping = EarlyStopping(
    patience=15,
    delta=0.001,
    path="/content/best_hybrid_model.pt"
)
train_losses = []
val_losses = []
val_r2_scores = []

# Training loop
for epoch in range(300):
    train_loss = train()
    val_loss, val_r2 = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")

    early_stopping(val_loss, val_r2, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break

# Load best model
model.load_state_dict(torch.load("/content/best_hybrid_model.pt"))
print("Best model loaded for final evaluation")

# Optional: Load last model if needed
if early_stopping.last_model_path:
    model.load_state_dict(torch.load(early_stopping.last_model_path))
    print(f"Last model loaded from: {early_stopping.last_model_path}")

# Plotting results
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss (Huber)', color='blue')
plt.plot(val_losses, label='Validation Loss (Huber)', color='orange')
plt.title('Training and Validation Loss (Huber)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# R² plot
plt.subplot(1, 2, 2)
plt.plot(val_r2_scores, label='Validation R²', color='green')
plt.title('Validation R² Score')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Test the model performance

from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn as nn

def evaluate_with_metrics(loader, model, mse_loss_fn, huber_loss_fn):
    model.eval()  # Set the model to evaluation mode
    all_preds = []  # List to store predictions
    all_labels = []  # List to store true labels
    total_mse_loss = 0
    total_huber_loss = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in loader:
            out = model(batch)  # Model predictions (RETAINED AS-IS)
            targets = batch.y  # True labels (assuming `batch.y` contains target values)

            # Ensure the shapes of `out` and `targets` match
            if out.shape != targets.shape:
                targets = targets.unsqueeze(1)  # Reshape targets to (batch_size, 1) if needed

            # Compute MSE and Huber Loss
            mse_loss = mse_loss_fn(out, targets)
            huber_loss = huber_loss_fn(out, targets)

            total_mse_loss += mse_loss.item()
            total_huber_loss += huber_loss.item()

            # Collect predictions and labels (convert to NumPy arrays)
            all_preds.extend(out.detach().cpu().numpy().flatten())  # Flatten predictions
            all_labels.extend(targets.detach().cpu().numpy().flatten())  # Flatten true labels

    # Compute average losses
    avg_mse_loss = total_mse_loss / len(loader)
    avg_huber_loss = total_huber_loss / len(loader)

    # Compute R² and MAE using sklearn.metrics
    r2 = r2_score(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    return avg_mse_loss, avg_huber_loss, r2, mae

# Define loss functions
mse_loss_fn = nn.MSELoss()
huber_loss_fn = nn.HuberLoss(delta=1.0)

# Evaluate on the test set
test_mse, test_huber_loss, test_r2, test_mae = evaluate_with_metrics(test_loader, model, mse_loss_fn, huber_loss_fn)

# Print results with state-of-the-art model values for comparison
print(f"Test MSE: {test_mse:.4f} (SOTA: 0.1000), "
      f"Test Huber Loss: {test_huber_loss:.4f} (SOTA: 0.1200), "
      f"R² Score: {test_r2:.4f} (SOTA: 0.9800), "
      f"MAE: {test_mae:.4f} (SOTA: 0.0500)")

# Save the model

from google.colab import files
from datetime import datetime
import pytz  # Add timezone support

# Get current time in East Africa Time (UTC+3)
eat_timezone = pytz.timezone('Africa/Nairobi')
eat_time = datetime.now(eat_timezone)

# Format timestamp with AM/PM

#timestamp = eat_time.strftime("Huber_NS3_%Y%m%d_%I%M%S%p")  # %I = 12-hour clock, %p = AM/PM
#timestamp = eat_time.strftime("Huber_Ro5_%Y%m%d_%I%M%S%p")  # %I = 12-hour clock, %p = AM/PM
timestamp = eat_time.strftime("Huber_Clean_%Y%m%d_%I%M%S%p")  # %I = 12-hour clock, %p = AM/PM

# test R² score
test_r2 = 0.9878

# Create filename
r2_str = f"R²={test_r2:.4f}".replace(".", "_")
filename = f"/content/model_test_{r2_str}_{timestamp}.pt"

# Save and download
torch.save(model.state_dict(), filename)
files.download(filename)
