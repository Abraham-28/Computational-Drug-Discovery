# -*- coding: utf-8 -*-
"""CDD_Part_6(Virtual Screening & Drug Designing).ipynb

# Refer my google colab link for notebook file https://colab.research.google.com/drive/1ORudo5SM1dHw16L-6tOS7lL53q8CLYjJ#scrollTo=dmtakZnOSj1S

## **Analyze a Drug Library**
## **Recap**

We've trained a `HybridGCNGAT` models using NS3 inhibitors from ChEMBL to predict pIC50 values from SMILES.
the next steps involve applying this model to identify potential NS3 inhibitors
for Dengue Virus Serotype 2 (DENV-2).




#### Step 1: Load the Drug Library
"""

!pip install torch torch-geometric rdkit pandas numpy

# Load the drug library dataset

import pandas as pd

drug_library_df = pd.read_csv("/content/5_Drug_molecules_library.csv")

# Checking if the dataset contains a 'smiles' column
if 'SMILES' not in drug_library_df.columns:
    raise ValueError("The dataset must contain a 'SMILES' column.")

print(f"Number of compounds in the drug library: {len(drug_library_df)}")

"""
#### Step 2: Preprocess the Drug Library
Convert each SMILES string into a PyTorch Geometric `Data` object."""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return None

    # Node features (atom-level)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),     # Degree of the atom
            atom.GetHybridization().real,  # Hybridization type
            atom.GetIsAromatic()  # Is the atom aromatic?
        ]
        node_features.append(features)

    # Edge indices (bond-level)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edges

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

# Convert each row into a graph representation
graphs = []
for _, row in drug_library_df.iterrows():
    data = smiles_to_graph(row["SMILES"])
    if data is not None:
        graphs.append(data)

print(f"Number of valid graphs in the drug library: {len(graphs)}")

"""#### Step 3: Create DataLoader for Prediction
Using PyTorch Geometric's `DataLoader` to create a data loader for the drug library.

"""

from torch_geometric.loader import DataLoader

# Define batch size
batch_size = 64

# Create data loader for the drug library
drug_library_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

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

"""#### Step 4: Predict pIC50 Values
Load the best trained `HybridGCNGAT` model and use it to predict pIC50 values for the drug library.
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# Load the trained model
model = HybridGCNGAT(hidden_dim=128, gat_heads=4)
model.load_state_dict(torch.load("/content/model_test_R²=0_9938_MAE_Clean_20250502_053709PM.pt", map_location='cpu'))


model.eval()

# Predict pIC50 values for the drug library
all_predictions = []

with torch.no_grad():
    for batch in drug_library_loader:
        out = model(batch)  # Get predictions
        all_predictions.extend(out.cpu().numpy())

# Add predictions to the drug library DataFrame
drug_library_df['predicted_pIC50'] = all_predictions

"""#### Step 5: Rank Compounds by Predicted pIC50
Sort the compounds by their predicted pIC50 values and extract the top candidates.
"""

# Sort by predicted pIC50 in descending order
ranked_compounds = drug_library_df.sort_values(by='predicted_pIC50', ascending=False)

# Extract the top 10 compounds
top_10_compounds = ranked_compounds.head(10)

print("Top 10 Compounds by Predicted pIC50:")
(top_10_compounds[['Drug_Name','SMILES', 'predicted_pIC50']])

# Save the Screened Drug compounds
ranked_compounds[['Drug_Name','SMILES', 'predicted_pIC50']].to_csv(
    '/content/Screened_Drug_compounds.csv',
    index=False
)
print("CSV file saved to /Screened_Drug_compounds.csv")

"""## **Generate New Molecules Using VGAE**

Define and implement a Variational Graph Autoencoder (VGAE) for the drug design stage. VGAE is particularly useful for learning latent representations of graphs, which can then be used for drug designing by generating drug molecules.


"""

# Step 1: Import Required Libraries

import torch
from torch.nn import Linear
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Define the Encoder and Decoder

# The VGAE model consists of an encoder and a decoder.
# The encoder learns the latent representation of the graph, and the decoder reconstructs the adjacency matrix.


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# Step 3: Load the dataset

from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atom-level)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),      # Atomic number
            atom.GetDegree(),         # Degree of the atom
            atom.GetHybridization().real,  # Hybridization type
            atom.GetIsAromatic()      # Is the atom aromatic?
        ]
        node_features.append(features)

    # Edge indices (bond-level)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edges

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)
    import pandas as pd

# Load dataset
df = pd.read_csv("/content/4-Clean_NS3_data (1)_augmented.csv")

# Keep only SMILES and pIC50 columns
df = df[["smiles", "pIC50"]].dropna()

# Convert SMILES to graphs
graphs = []
for _, row in df.iterrows():
    data = smiles_to_graph(row["smiles"], row["pIC50"])
    if data is not None:
        graphs.append(data)

print(f"Number of valid graphs: {len(graphs)}")

# Step 4: Split the dataset using k fold

from sklearn.model_selection import KFold

# K-Fold split (5 folds)
def kfold_split(graphs, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
        yield [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]

for fold, (train_data, val_data) in enumerate(kfold_split(graphs)):
    print(f"Fold {fold + 1}: Train={len(train_data)}, Val={len(val_data)}")

    # Create DataLoaders for the current fold
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Step 5: Define the VGAE Model

from torch_geometric.nn import VGAE, GCNConv
from torch.nn import BatchNorm1d, Linear

# Enhanced Encoder
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # Add batch normalization
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)  # Apply batch normalization
        x = self.conv2(x, edge_index).relu()
        mu = self.conv2(x, edge_index)
        logvar = self.conv3(x, edge_index)
        return mu, logvar

# Enhanced Decoder
class MLPDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = Linear(latent_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, latent_dim)

    def forward(self, z, sigmoid=True):
        adj = torch.matmul(self.fc2(self.fc1(z).relu()), z.t())
        return torch.sigmoid(adj) if sigmoid else adj

# Initialize model components
in_channels = graphs[0].x.shape[1]
encoder = VariationalGCNEncoder(in_channels, hidden_channels=256, out_channels=128)
decoder = MLPDecoder(latent_dim=128, hidden_dim=256)

# Create VGAE model
vgae_model = VGAE(encoder, decoder).to('cuda' if torch.cuda.is_available() else 'cpu')

# Step 6: Train the VGAE Model

import torch
from torch_geometric.nn import VGAE, GCNConv
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datetime import datetime
import os
import matplotlib.pyplot as plt

# --- Early Stopping with Timestamped Checkpoints ---
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_path = None

    def __call__(self, val_loss, model):
        score = -val_loss
        prev_loss = self.best_val_loss

        if self.best_score is None:
            self.save_checkpoint(prev_loss, val_loss, model)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.save_checkpoint(prev_loss, val_loss, model)
            self.best_score = score
            self.counter = 0
        return False

    def save_checkpoint(self, prev_loss, current_loss, model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vgae_val-loss_tm1_{current_loss:.4f}_{timestamp}.pt"
        torch.save(model.state_dict(), filename)

        # Keep only the best model
        if self.best_model_path:
            try: os.remove(self.best_model_path)
            except: pass

        self.best_model_path = filename
        self.best_val_loss = current_loss

        if prev_loss == float('inf'):
            print(f". Initial model saved: {filename}")
        else:
            print(f"  Validation loss decreased ({prev_loss:.4f} → {current_loss:.4f}). New best: {filename}")

# --- VGAE Model Components ---
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index=None, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, latent_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

# --- Training Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# gives the number of input features
in_channels = graphs[0].x.shape[1]

# Initialize encoder and decoder with latent dimension
encoder = VariationalGCNEncoder(in_channels=in_channels, latent_dim=128).to(device)
decoder = InnerProductDecoder().to(device)
model = VGAE(encoder, decoder).to(device)

optimizer = Adam(model.parameters(), lr=0.0001)
early_stopping = EarlyStopping(patience=15)

# --- Training Loop ---
train_losses = []
val_losses = []

for epoch in range(300):
    # Train phase
    model.train()
    total_train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index) + model.kl_loss()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index) + model.kl_loss()
            total_val_loss += loss.item()
    val_loss = total_val_loss / len(val_loader)
    val_losses.append(val_loss)

    # Progress and checkpointing
    print(f"\rEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", end='')
    if early_stopping(val_loss, model):
        print("\nEarly stopping triggered")
        break

# --- Load Best Model ---
if early_stopping.best_model_path:
    model.load_state_dict(torch.load(early_stopping.best_model_path))
    print(f"\nLoaded best model: {early_stopping.best_model_path}")
else:
    print("No checkpoints saved - using last model state")

# --- Plot Results ---
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Step 7: Generate latent space from the model

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv

# Automatically detect input dimension from checkpoint
def get_input_dim_from_checkpoint(checkpoint_path):
    """
    Detects the input dimension from the checkpoint's state dictionary.
    Inspects the first layer's weight shape to determine the input dimension.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    print("Checkpoint keys:")
    for key in state_dict.keys():
        print(key)

    # Find the first layer's weight shape
    for key in state_dict:
        if 'conv1.lin.weight' in key:
            return state_dict[key].shape[1]
    raise ValueError("Could not determine input dimension from checkpoint")

# Detect input dimension
checkpoint_path = "/content/vgae_val-loss_tm1_1.7776_20250518_203356.pt"
in_channels = get_input_dim_from_checkpoint(checkpoint_path)

#  Model Architecture
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, latent_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

# Load Trained Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model with detected input dimension
latent_dim = 128
encoder = VariationalGCNEncoder(in_channels=in_channels, latent_dim=latent_dim).to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Strip "encoder." prefix from state_dict keys
stripped_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}

# Load stripped state_dict into the encoder
encoder.load_state_dict(stripped_state_dict, strict=False)
encoder.eval()

# Convert SMILES to Graphs

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atomic number, degree, formal charge, hybridization)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real
        ])
    x = torch.tensor(atom_features, dtype=torch.float)  # Shape: [num_nodes, 4]

    # Edge indices
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data

# Encode Molecules into Latent Space
def encode_to_latent(model, loader, device):
    """
    Encodes molecules into their latent space representations.
    Returns a list of latent vectors (mu) for each molecule.
    """
    latent_vectors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model(batch.x, batch.edge_index)  # Get mu (mean of latent space)

            # Aggregate node embeddings for each molecule in the batch
            latent_vectors.extend([mu[batch.batch == i].mean(dim=0).cpu().numpy() for i in range(batch.num_graphs)])
    return latent_vectors


# Load dataset
data_path = "/content/4-Clean_NS3_data (1)_augmented.csv"
df = pd.read_csv(data_path)

# Convert SMILES to graphs
graphs = []
for smiles in df['smiles']:
    data = smiles_to_graph(smiles)
    if data is not None:
        graphs.append(data)

# Create DataLoader
batch_size = 32
loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

# Generate latent space
latent_vectors = encode_to_latent(encoder, loader, device)

# Save latent vectors to a DataFrame
latent_df = pd.DataFrame(latent_vectors, columns=[f"latent_{i}" for i in range(latent_dim)])
latent_df['smiles'] = df['smiles'][:len(latent_vectors)]  # Add SMILES for reference

# Save to CSV
latent_df.to_csv("latent_space.csv", index=False)

print("Latent space saved to 'latent_space.csv'")

# Step 8: Load the latent space and generate molecules

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Load Latent Space
latent_space_path = "/content/latent_space.csv"
latent_df = pd.read_csv(latent_space_path)

# Extract latent vectors and SMILES
latent_vectors = latent_df[[f"latent_{i}" for i in range(64)]].values  # Adjust based on latent_dim
smiles_list = latent_df['smiles'].values

# Define New Decoder
class FullyConnectedDecoder(torch.nn.Module):
    def __init__(self, latent_dim, max_nodes):
        super().__init__()
        self.fc_adj = torch.nn.Linear(latent_dim, max_nodes * max_nodes)  # For adjacency matrix
        self.fc_atom = torch.nn.Linear(latent_dim, max_nodes * 5)         # For atom features (5 possible atom types)

    def forward(self, z, sigmoid=True):
        # Predict adjacency matrix
        adj_flat = self.fc_adj(z)
        adj = adj_flat.view(-1, max_nodes, max_nodes)  # Reshape to [batch_size, max_nodes, max_nodes]

        # Predict atom features
        atom_flat = self.fc_atom(z)
        atom_features = atom_flat.view(-1, max_nodes, 5)  # Reshape to [batch_size, max_nodes, num_atom_types]

        return torch.sigmoid(adj) if sigmoid else adj, atom_features

# Initialize decoder
max_nodes = 38
decoder = FullyConnectedDecoder(latent_dim=64, max_nodes=max_nodes)

# Sample New Latent Vectors
def sample_latent_space(latent_vectors, num_samples=50):
    """
    Samples new latent vectors by interpolating between existing latent vectors.
    """
    latent_dim = latent_vectors.shape[1]
    sampled_latents = []

    for _ in range(num_samples):
        # Randomly select two latent vectors
        idx1, idx2 = np.random.choice(len(latent_vectors), size=2, replace=False)
        vec1, vec2 = latent_vectors[idx1], latent_vectors[idx2]

        # Interpolate between the two vectors
        alpha = np.random.uniform(0, 1)
        sampled_latent = alpha * vec1 + (1 - alpha) * vec2
        sampled_latents.append(sampled_latent)

    return np.array(sampled_latents)

# Sample 50 new latent vectors
sampled_latents = sample_latent_space(latent_vectors, num_samples=50)

# --- Decode Latent Vectors into Molecular Graphs ---
def decode_to_graphs(decoder, latents, max_nodes=38, threshold=0.5):
    """
    Decodes latent vectors into molecular graphs using the VGAE decoder.
    """
    graphs = []
    with torch.no_grad():
        for latent in latents:
            # Reshape latent vector to match decoder expectations
            z = torch.tensor(latent, dtype=torch.float).unsqueeze(0)  # Shape: [1, latent_dim]

            # Predict adjacency matrix and atom features
            adj, atom_features = decoder(z, sigmoid=True)
            adj = adj.squeeze().cpu().numpy()
            atom_features = atom_features.squeeze().cpu().numpy()

            # Ensure adjacency matrix has the correct shape
            if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
                print(f"Invalid adjacency matrix shape: {adj.shape}. Skipping...")
                continue

            # Create molecule from adjacency matrix and atom features
            mol = Chem.RWMol()
            atom_valences = np.zeros(max_nodes, dtype=int)
            atom_types = []  # Track atom types

            # Add atoms based on predicted atom features
            for i in range(max_nodes):
                atom_type = np.argmax(atom_features[i])  # Predicted atom type (0: C, 1: N, 2: O, 3: F, 4: H)
                if atom_type == 0:
                    mol.AddAtom(Chem.Atom(6))  # Carbon
                    atom_types.append(4)       # Max valence for carbon
                elif atom_type == 1:
                    mol.AddAtom(Chem.Atom(7))  # Nitrogen
                    atom_types.append(3)       # Max valence for nitrogen
                elif atom_type == 2:
                    mol.AddAtom(Chem.Atom(8))  # Oxygen
                    atom_types.append(2)       # Max valence for oxygen
                elif atom_type == 3:
                    mol.AddAtom(Chem.Atom(9))  # Fluorine
                    atom_types.append(1)       # Max valence for fluorine
                elif atom_type == 4:
                    mol.AddAtom(Chem.Atom(1))  # Hydrogen
                    atom_types.append(1)       # Max valence for hydrogen

            # Add bonds while respecting valence
            for i in range(max_nodes):
                for j in range(i + 1, max_nodes):
                    if adj[i, j] > threshold:
                        # Check valence constraints
                        if atom_valences[i] < atom_types[i] and atom_valences[j] < atom_types[j]:
                            try:
                                mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                                atom_valences[i] += 1
                                atom_valences[j] += 1
                            except Exception as e:
                                print(f"Failed to add bond: {e}")
                                continue

            # Remove isolated hydrogen atoms
            mol = mol.GetMol()
            mol = Chem.RemoveHs(mol)  # Remove hydrogens without neighbors

            # Final validation and post-processing
            try:
                # Sanitize molecule
                Chem.SanitizeMol(mol)

                # Assign aromaticity and kekulize
                Chem.Kekulize(mol, clearAromaticFlags=True)

                # Check for basic validity
                if mol.GetNumAtoms() > 1 and mol.GetNumBonds() > 0:
                    graphs.append(mol)
            except Exception as e:
                print(f"Failed to create molecule: {e}")
                continue

    return graphs

# Decode sampled latent vectors into molecular graphs
decoded_graphs = decode_to_graphs(decoder, sampled_latents, max_nodes=38, threshold=0.5)

# Convert Graphs to SMILES
generated_smiles = [Chem.MolToSmiles(mol) for mol in decoded_graphs]

# Save Generated SMILES to .smi File
smi_file_path = "/content/generated_molecules.smi"
with open(smi_file_path, "w") as f:
    for smiles in generated_smiles:
        f.write(smiles + "\n")

print(f"Generated SMILES saved to '{smi_file_path}'.")

# Visualize Generated Molecules
def plot_molecules(mols, mols_per_row=5, sub_img_size=(200, 200)):
    """
    Plots the structures of the generated molecules.
    """
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        legends=[Chem.MolToSmiles(mol) for mol in mols],
        subImgSize=sub_img_size
    )
    return img

# Plot the top 50 generated molecules
generated_mols = [Chem.MolFromSmiles(smiles) for smiles in generated_smiles]

if len(generated_mols) == 0:
    print("No valid molecules were generated. Skipping visualization.")
else:
    display(plot_molecules(generated_mols))

# Step 9: Save the generated molecules in csv format
import pandas as pd

# Path to the .smi file
smi_file_path = "/content/generated_molecules.smi"

# Path to save the CSV file
csv_file_path = "/content/generated_molecules.csv"

# Read the SMILES strings from the .smi file
with open(smi_file_path, "r") as f:
    smiles_list = [line.strip() for line in f if line.strip()]

# Create a DataFrame with the SMILES strings
df = pd.DataFrame(smiles_list, columns=["SMILES"])

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Generated molecules saved to '{csv_file_path}' in CSV format.")

# Step 10: Load the generated molecules library

import pandas as pd

drug_library_df = pd.read_csv("/content/generated_molecules.csv")


# Ensure the dataset contains a 'smiles' column
if 'SMILES' not in drug_library_df.columns:
    raise ValueError("The dataset must contain a 'SMILES' column.")

print(f"Number of compounds in the drug library: {len(drug_library_df)}")

# Step 11: Preprocess the Library

# Convert each SMILES string into a PyTorch Geometric Data object using


import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return None

    # Node features (atom-level)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),     # Degree of the atom
            atom.GetHybridization().real,  # Hybridization type
            atom.GetIsAromatic()  # Is the atom aromatic?
        ]
        node_features.append(features)

    # Edge indices (bond-level)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edges

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

# Convert each row into a graph representation
graphs = []
for _, row in drug_library_df.iterrows():
    data = smiles_to_graph(row["SMILES"])
    if data is not None:
        graphs.append(data)

print(f"Number of valid graphs in the drug library: {len(graphs)}")

# Step 12: Create DataLoader for Prediction

#Use PyTorch Geometric's DataLoader to create a data loader for the drug library.

from torch_geometric.loader import DataLoader

# Define batch size
batch_size = 64

# Create data loader for the drug library
drug_library_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

# Step 13: Predict pIC50 Values

# Load the best trained HybridGCNGAT model to predict pIC50 values for the drug library.

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# Load the trained model
model = HybridGCNGAT(hidden_dim=128, gat_heads=4)
model.load_state_dict(torch.load("/content/model_test_R²=0_9971_MAE_Clean_20250507_045143PM.pt", map_location='cpu'))
model.eval()

# Predict pIC50 values for the drug library
all_predictions = []

with torch.no_grad():
    for batch in drug_library_loader:
        out = model(batch)  # Get predictions
        all_predictions.extend(out.cpu().numpy())

# Add predictions to the drug library DataFrame
drug_library_df['predicted_pIC50'] = all_predictions

# Step 14: Rank Compounds by Predicted pIC50

#Sort the compounds by their predicted pIC50 values and extract the top candidates.

# Sort by predicted pIC50 in descending order
ranked_compounds = drug_library_df.sort_values(by='predicted_pIC50', ascending=False)

# Extract the top 10 compounds
top_10_compounds = ranked_compounds.head(10)

print("Top 10 Compounds by Predicted pIC50:")
#(top_10_compounds[['Drug_Name','SMILES', 'predicted_pIC50']])
(top_10_compounds[['SMILES', 'predicted_pIC50']])

# Step 14: Rank Compounds by Predicted pIC50 and Structural Uniqueness Check

from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import matplotlib.pyplot as plt

def get_morgan_fingerprint(smiles):
    """Generate Morgan fingerprint for similarity checking"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprint(mol, 2)
    return None

# Sort compounds and remove duplicates
ranked_compounds = drug_library_df.sort_values(by='predicted_pIC50', ascending=False)

# Deduplication based on structural similarity
seen_fingerprints = []
unique_compounds = []

for _, row in ranked_compounds.iterrows():
    fp = get_morgan_fingerprint(row['SMILES'])
    if not fp:
        continue

    # Check against existing compounds
    similarity = [DataStructs.TanimotoSimilarity(fp, seen_fp) for seen_fp in seen_fingerprints]
    if not any(sim > 0.7 for sim in similarity):  # 0.7 similarity threshold
        seen_fingerprints.append(fp)
        unique_compounds.append(row)

    if len(unique_compounds) >= 10:
        break

# Create DataFrame from unique compounds
top_10_unique = pd.DataFrame(unique_compounds).reset_index(drop=True)

print("Unique Top 10 Compounds by Predicted pIC50:")
print(top_10_unique[['SMILES', 'predicted_pIC50']])

# Generate 2D structure images
mols = [Chem.MolFromSmiles(smi) for smi in top_10_unique['SMILES']]
legends = [f"pIC50: {pIC50:.2f}" for pIC50 in top_10_unique['predicted_pIC50']]

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(200, 200),
    legends=legends,
    returnPNG=False
)

# Display structures
plt.figure(figsize=(15, 6))
plt.imshow(img)
plt.axis('off')
plt.show()

# Save visualization
img.save("top10_structures.png")
print("\n2D structures saved to top10_structures.png")

# Save the top unique compounds
top_10_unique[['SMILES', 'predicted_pIC50']].to_csv(
    '/content/top_10_unique_compounds.csv',
    index=False
)
print("CSV file saved to /content/top_10_unique_compounds.csv")
