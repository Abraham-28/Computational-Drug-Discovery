# -*- coding: utf-8 -*-
"""CDD_Part_4(Augmentation).ipynb

# Refer my google colab link for notebook file https://colab.research.google.com/drive/1htFOiDtAisaIfaBJcryQ5ensePJL3eT-#scrollTo=z2-B8Ah5BH8z
#Augment 1-NS3_(1) dataset
"""

!pip install rdkit pandas numpy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Load original data
df = pd.read_csv("/content/1-NS3_(1).csv")

# Define valid SMILES for substituents (expanded for diversity)
SUBSTITUENTS = {
    "[Cl]": ["[F]", "[Br]", "[I]", "[C](F)(F)F"],  # Halogens + CF3
    "[O][C]": ["[O][C](F)(F)F", "[O][C]C", "[C](F)(F)F"],  # Methoxy → Trifluoromethoxy/Ethoxy/CF3
    "[C](=O)[O]": ["[C](=O)[N]", "[C](=O)[S]", "[C](=O)[O]C"],  # Carboxylic acid derivatives
    "c1ccccc1": ["c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1"],  # Aromatic ring substitutions
}

def augment_smiles(smiles, num_variants=300):
    """Generate new SMILES via substructure replacement with validity checks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    new_smiles = []
    for _ in range(num_variants):
        mol_copy = Chem.Mol(mol)
        replaced = False

        # Substructure Replacement
        for frag in SUBSTITUENTS:
            query = Chem.MolFromSmarts(frag)
            if not query:
                continue

            if mol_copy.HasSubstructMatch(query):
                replacement = np.random.choice(SUBSTITUENTS[frag])
                replacement_mol = Chem.MolFromSmarts(replacement)
                if replacement_mol:
                    new_mol = AllChem.ReplaceSubstructs(
                        mol_copy, query, replacement_mol, replaceAll=True
                    )
                    if new_mol:
                        new_mol = new_mol[0]
                        new_smiles.append(Chem.MolToSmiles(new_mol))
                        replaced = True
                        break

        if not replaced:
            # Fallback: Use SMILES Enumeration (Randomize Atom Order)
            randomized_smiles = Chem.MolToSmiles(mol_copy, canonical=False, doRandom=True)
            new_smiles.append(randomized_smiles)

    return list(set(new_smiles))

def calculate_lipinski_descriptors(smiles):
    """Calculate Lipinski descriptors to ensure drug-likeness."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.ExactMolWt(mol)  # Molecular Weight
    logp = Descriptors.MolLogP(mol)   # LogP
    hbd = rdMolDescriptors.CalcNumHBD(mol)  # Hydrogen Bond Donors
    hba = rdMolDescriptors.CalcNumHBA(mol)  # Hydrogen Bond Acceptors
    tpsa = Descriptors.TPSA(mol)      # Topological Polar Surface Area
    rot_bonds = Descriptors.NumRotatableBonds(mol)  # Rotatable Bonds

    # Check stricter drug-likeness criteria
    if (
        mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and
        tpsa <= 140 and rot_bonds <= 10
    ):
        return True
    return False

# Generate augmented rows
augmented_data = []
for _, row in df.iterrows():
    variants = augment_smiles(row["smiles"], num_variants=5)
    for variant in variants:
        # Ensure chemical validity and drug-likeness
        if Chem.MolFromSmiles(variant) and calculate_lipinski_descriptors(variant):
            new_pIC50 = row["pIC50"] + np.random.normal(0, 0.1)  # Add noise to pIC50
            augmented_data.append({
                "smiles": variant,
                "pIC50": np.clip(new_pIC50, 3.0, 9.0)  # Ensure realistic range
            })

# Save to CSV
augmented_df = pd.DataFrame(augmented_data)
augmented_df = augmented_df.drop_duplicates(subset=["smiles"])
augmented_df = augmented_df.sample(n=5000, replace=True).reset_index(drop=True)
augmented_df.to_csv("4-NS3_(1)_augmented.csv", index=False)

print("Augmented dataset saved to '4-NS3_(1)_augmented.csv'")

# --- Visualize Unique Molecules ---
unique_smiles = augmented_df["smiles"].unique()[:50]  # Visualize the first 50 unique molecules
mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]

# Check for invalid molecules
valid_mols = [mol for mol in mols if mol is not None]

# Draw the molecules
img = Draw.MolsToGridImage(
    valid_mols,
    legends=[smiles for smiles in unique_smiles if Chem.MolFromSmiles(smiles)],
    molsPerRow=5,
    subImgSize=(200, 200),
    useSVG=True
)

# Display the image
display(img)

"""#Augment ro5_properties_fulfilled dataset"""

!pip install rdkit pandas numpy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Load original data
df = pd.read_csv("/content/2-ro5_properties_fulfilled.csv")

# Define valid SMILES for substituents (expanded for diversity)
SUBSTITUENTS = {
    "[Cl]": ["[F]", "[Br]", "[I]", "[C](F)(F)F"],  # Halogens + CF3
    "[O][C]": ["[O][C](F)(F)F", "[O][C]C", "[C](F)(F)F"],  # Methoxy → Trifluoromethoxy/Ethoxy/CF3
    "[C](=O)[O]": ["[C](=O)[N]", "[C](=O)[S]", "[C](=O)[O]C"],  # Carboxylic acid derivatives
    "c1ccccc1": ["c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1"],  # Aromatic ring substitutions
}

def augment_smiles(smiles, num_variants=5):
    """Generate new SMILES via substructure replacement with validity checks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    new_smiles = []
    for _ in range(num_variants):
        mol_copy = Chem.Mol(mol)
        replaced = False

        # Substructure Replacement
        for frag in SUBSTITUENTS:
            query = Chem.MolFromSmarts(frag)
            if not query:
                continue

            if mol_copy.HasSubstructMatch(query):
                replacement = np.random.choice(SUBSTITUENTS[frag])
                replacement_mol = Chem.MolFromSmarts(replacement)
                if replacement_mol:
                    new_mol = AllChem.ReplaceSubstructs(
                        mol_copy, query, replacement_mol, replaceAll=True
                    )
                    if new_mol:
                        new_mol = new_mol[0]
                        new_smiles.append(Chem.MolToSmiles(new_mol))
                        replaced = True
                        break

        if not replaced:
            # Fallback: Use SMILES Enumeration (Randomize Atom Order)
            randomized_smiles = Chem.MolToSmiles(mol_copy, canonical=False, doRandom=True)
            new_smiles.append(randomized_smiles)

    return list(set(new_smiles))

def calculate_lipinski_descriptors(smiles):
    """Calculate Lipinski descriptors to ensure drug-likeness."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.ExactMolWt(mol)  # Molecular Weight
    logp = Descriptors.MolLogP(mol)   # LogP
    hbd = rdMolDescriptors.CalcNumHBD(mol)  # Hydrogen Bond Donors
    hba = rdMolDescriptors.CalcNumHBA(mol)  # Hydrogen Bond Acceptors
    tpsa = Descriptors.TPSA(mol)      # Topological Polar Surface Area
    rot_bonds = Descriptors.NumRotatableBonds(mol)  # Rotatable Bonds

    # Check stricter drug-likeness criteria
    if (
        mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and
        tpsa <= 140 and rot_bonds <= 10
    ):
        return True
    return False

# Generate augmented rows
augmented_data = []
for _, row in df.iterrows():
    variants = augment_smiles(row["smiles"], num_variants=5)
    for variant in variants:
        # Ensure chemical validity and drug-likeness
        if Chem.MolFromSmiles(variant) and calculate_lipinski_descriptors(variant):
            new_pIC50 = row["pIC50"] + np.random.normal(0, 0.1)  # Add noise to pIC50
            augmented_data.append({
                "smiles": variant,
                "pIC50": np.clip(new_pIC50, 3.0, 9.0)  # Ensure realistic range
            })

# Save to CSV
augmented_df = pd.DataFrame(augmented_data)
augmented_df = augmented_df.drop_duplicates(subset=["smiles"])
augmented_df = augmented_df.sample(n=5000, replace=True).reset_index(drop=True)
augmented_df.to_csv("4-ro5_properties_fulfilled_augmented.csv", index=False)

print("Augmented dataset saved to '4-ro5_properties_fulfilled_augmented.csv'")

# --- Visualize Unique Molecules ---
unique_smiles = augmented_df["smiles"].unique()[:20]  # Visualize the first 20 unique molecules
mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]

# Check for invalid molecules
valid_mols = [mol for mol in mols if mol is not None]

# Draw the molecules
img = Draw.MolsToGridImage(
    valid_mols,
    legends=[smiles for smiles in unique_smiles if Chem.MolFromSmiles(smiles)],
    molsPerRow=5,
    subImgSize=(200, 200),
    useSVG=True
)

# Display the image
display(img)

"""#Augument Clean_NS3_data (1) dataset"""

!pip install rdkit pandas numpy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Load original data
df = pd.read_csv("/content/3-Clean_NS3_data (1).csv")

# Define valid SMILES for substituents (for diversity)
SUBSTITUENTS = {
    "[Cl]": ["[F]", "[Br]", "[I]", "[C](F)(F)F"],  # Halogens + CF3
    "[O][C]": ["[O][C](F)(F)F", "[O][C]C", "[C](F)(F)F"],  # Methoxy → Trifluoromethoxy/Ethoxy/CF3
    "[C](=O)[O]": ["[C](=O)[N]", "[C](=O)[S]", "[C](=O)[O]C"],  # Carboxylic acid derivatives
    "c1ccccc1": ["c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1"],  # Aromatic ring substitutions
}

def augment_smiles(smiles, num_variants=200):
    """Generate new SMILES via substructure replacement with validity checks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    new_smiles = []
    for _ in range(num_variants):
        mol_copy = Chem.Mol(mol)
        replaced = False

        # Substructure Replacement
        for frag in SUBSTITUENTS:
            query = Chem.MolFromSmarts(frag)
            if not query:
                continue

            if mol_copy.HasSubstructMatch(query):
                replacement = np.random.choice(SUBSTITUENTS[frag])
                replacement_mol = Chem.MolFromSmarts(replacement)
                if replacement_mol:
                    new_mol = AllChem.ReplaceSubstructs(
                        mol_copy, query, replacement_mol, replaceAll=True
                    )
                    if new_mol:
                        new_mol = new_mol[0]
                        new_smiles.append(Chem.MolToSmiles(new_mol))
                        replaced = True
                        break

        if not replaced:
            # Fallback: Use SMILES Enumeration (Randomize Atom Order)
            randomized_smiles = Chem.MolToSmiles(mol_copy, canonical=False, doRandom=True)
            new_smiles.append(randomized_smiles)

    return list(set(new_smiles))

def calculate_lipinski_descriptors(smiles):
    """Calculate Lipinski descriptors to ensure drug-likeness."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.ExactMolWt(mol)  # Molecular Weight
    logp = Descriptors.MolLogP(mol)   # LogP
    hbd = rdMolDescriptors.CalcNumHBD(mol)  # Hydrogen Bond Donors
    hba = rdMolDescriptors.CalcNumHBA(mol)  # Hydrogen Bond Acceptors
    tpsa = Descriptors.TPSA(mol)      # Topological Polar Surface Area
    rot_bonds = Descriptors.NumRotatableBonds(mol)  # Rotatable Bonds

    # Check stricter drug-likeness criteria
    if (
        mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and
        tpsa <= 140 and rot_bonds <= 10
    ):
        return True
    return False

# Generate augmented rows
augmented_data = []
for _, row in df.iterrows():
    variants = augment_smiles(row["smiles"], num_variants=200)
    for variant in variants:
        # Ensure chemical validity and drug-likeness
        if Chem.MolFromSmiles(variant) and calculate_lipinski_descriptors(variant):
            new_pIC50 = row["pIC50"] + np.random.normal(0, 0.1)  # Add noise to pIC50
            augmented_data.append({
                "smiles": variant,
                "pIC50": np.clip(new_pIC50, 3.0, 9.0)  # Ensure realistic range
            })

# Save to CSV
augmented_df = pd.DataFrame(augmented_data)
augmented_df = augmented_df.drop_duplicates(subset=["smiles"])
augmented_df = augmented_df.sample(n=5000, replace=True).reset_index(drop=True)
augmented_df.to_csv("4-Clean_NS3_data (1)_augmented(2).csv", index=False)

print("Augmented dataset saved to '4-Clean_NS3_data (1)_augmented(2).csv'")

# --- Visualize Unique Molecules ---
unique_smiles = augmented_df["smiles"].unique()[:100]  # Visualize the first 100 unique molecules
mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]

# Check for invalid molecules
valid_mols = [mol for mol in mols if mol is not None]

# Draw the molecules
img = Draw.MolsToGridImage(
    valid_mols,
    legends=[smiles for smiles in unique_smiles if Chem.MolFromSmiles(smiles)],
    molsPerRow=5,
    subImgSize=(200, 200),
    useSVG=True
)

# Display the image
display(img)

# Count valid molecules
valid_count = 0
for smi in augmented_df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol and calculate_descriptors(smi):
        valid_count += 1
print(f"Valid molecules: {valid_count}/{len(augmented_df)}")

# Visualize unique molecules
unique_smiles = augmented_df['smiles'].unique()[:100]
mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles]
valid_mols = [mol for mol in mols if mol and calculate_descriptors(Chem.MolToSmiles(mol))]  # Pass SMILES string
img = MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(200, 200), useSVG=True)
display(img)
