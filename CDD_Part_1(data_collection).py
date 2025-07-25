# -*- coding: utf-8 -*-
"""CDD_Part_1(Data-Collection).ipynb

# Refer my google colab link for notebook file https://colab.research.google.com/drive/1XiTszdGkNQnFtapPutCLCl76agHgDfx-#scrollTo=SVc2uvlG7ifW
# **Data Retrieval From ChEMBL Database**

**Objectives**

Retrieving compound data and activities from ChEMBL.

###Pre-requisite Installation and Imports
"""

! pip install chembl_webresource_client
! pip install rdkit-pypi

import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm

"""###Define paths for working with files and directories."""

HERE = Path(_dh[-1])
DATA = HERE / "data"

"""Creating API tokens for target"""

targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity

type(targets_api)

""" Here we will use the **Dengue virus type 2 NS3 protein**, which is a single protein with enzymatic activity. We will fetch its information using the Uniprot Accession ID P29990, corresponding to the target protein CHEMBL5980. This protein is derived from Dengue virus type 2 (strain Thailand/16681/1984) (DENV-2)."""

uniprot_id = "P29990"

# Get target information from ChEMBL with specified values

targets = targets_api.get(target_components__accession=uniprot_id).only(
    "target_chembl_id", "organism", "pref_name", "target_type"
)
print(f'The type of the targets is "{type(targets)}"')

targets = pd.DataFrame.from_records(targets)
targets

"""Below we will select ChEMBL data via index options."""

target = targets.iloc[0]
target

# NBVAL_CHECK_OUTPUT

chembl_id = target.target_chembl_id
print(f"The target ChEMBL ID is {chembl_id}")

"""Then we fetch some important variable of the data."""

bioactivities = bioactivities_api.filter(
    target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B"
).only(
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "molecule_chembl_id",
    "type",
    "standard_units",
    "relation",
    "standard_value",
    "target_chembl_id",
    "target_organism",
)

print(f"Length and type of bioactivities object: {len(bioactivities)}, {type(bioactivities)}")

print(f"Length and type of first element: {len(bioactivities[0])}, {type(bioactivities[0])}")
bioactivities[0]

#Each entry in our bioactivity set holds the following information:

"""**IC50** is a standard unit used in biochemical assays for inhibition. It stands for Inhibitory concentration at 50 or Half maximal inhibitory concentration. The lower value indicates higher potency of a drug.

**pIC50**, that is = -log10(IC50). It is the opposite of IC50, meaning higher pIC50 means higher potency of a drug.

Download the dataset
"""

#Download the Dataset
bioactivities_df = pd.DataFrame.from_dict(bioactivities)
print(f"DataFrame shape: {bioactivities_df.shape}")
bioactivities_df.head()

#Create csv file for the dataset
bioactivities_df.to_csv('bioactivities.csv', index=False)

"""
**Now we want to know how many different units are present in the data**"""

bioactivities_df["units"].unique()

"""**Drop the last two columns to keep the standard units value for uniformity**"""

bioactivities_df.drop(["units", "value"], axis=1, inplace=True)
bioactivities_df.head()

"""**Data Processing and Filteration**"""

bioactivities_df.dtypes

# Change data type of "standard_vaue" to float

bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
bioactivities_df.dtypes

# df shape before dropping missing values

print(f"DataFrame shape: {bioactivities_df.shape}")

# df shape after dropping any missing values

bioactivities_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")

"""**Filter non-nM entries**"""

print(f"Units in downloaded data: {bioactivities_df['standard_units'].unique()}")
print(
    f"Number of non-nM entries:\
    {bioactivities_df[bioactivities_df['standard_units'] != 'nM'].shape[0]}"
)

bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
print(f"Units after filtering: {bioactivities_df['standard_units'].unique()}")
print(
    f"Number of non-nM entries:\
    {bioactivities_df[bioactivities_df['standard_units'] != 'nM'].shape[0]}"
)

print(f"DataFrame shape: {bioactivities_df.shape}")

"""**Drop duplicate rows**"""

bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")

bioactivities_df.reset_index(drop=True, inplace=True)
bioactivities_df.head()

# Rename some columns

bioactivities_df.rename(
    columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True
)
bioactivities_df.head()

print(f"DataFrame shape: {bioactivities_df.shape}")

"""**Fetch compound data from ChEMBL**

Let’s have a look at the compounds from ChEMBL which we have defined so far as bioactivity data. We are going to fetch compound ChEMBL IDs and structures for the compounds linked to our filtered bioactivity data.
"""

compounds_provider = compounds_api.filter(
    molecule_chembl_id__in=list(bioactivities_df["molecule_chembl_id"])
).only("molecule_chembl_id", "molecule_structures")

"""**Download Compounds**"""

compounds = list(tqdm(compounds_provider))

compounds_df = pd.DataFrame.from_records(
    compounds,
)
print(f"DataFrame shape: {compounds_df.shape}")

compounds_df.head()

compounds_df.iloc[0].molecule_structures.keys()

#The keys are:

#canonical_smiles → SMILES string format (most commonly used for structures).

#molfile → Structure in MDL Molfile format.

#standard_inchi → Standardized InChI string (International Chemical Identifier).

#standard_inchi_key → Hashed InChI key (fixed-length string).

"""**Extract the SMILES from the dictionary**"""

canonical_smiles = []

for i, compounds in compounds_df.iterrows():
    try:
        canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
    except KeyError:
        canonical_smiles.append(None)

compounds_df["smiles"] = canonical_smiles
compounds_df.drop("molecule_structures", axis=1, inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")

compounds_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")

compounds_df.head()

"""**Let's output compound bioactivity data**"""

print(f"Bioactivities filtered: {bioactivities_df.shape[0]}")
bioactivities_df.columns

print(f"Compounds filtered: {compounds_df.shape[0]}")
compounds_df.columns

"""**Let's merge both the datasets**

which will contain, molecule ID, IC50, Units, and canonical smiles
"""

# Merge DataFrames
output_df = pd.merge(
    bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
    compounds_df,
    on="molecule_chembl_id",
)

# Reset row indices
output_df.reset_index(drop=True, inplace=True)

print(f"Dataset with {output_df.shape[0]} entries.")

output_df.dtypes

output_df.head(192)

"""**Convert IC50 values into pIC50 and include it in new column**"""

def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value

# Apply conversion to each row of the compounds DataFrame
output_df["pIC50"] = output_df.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)

output_df.head(192)

"""**Distribution Based on pIC50**"""

output_df.hist(column="pIC50")

"""**Now we add a column for RDKit molecule objects to our DataFrame and look at the structures of the molecules with their pIC50 values.**"""

import pandas as pd
from rdkit.Chem import PandasTools

# Add molecule column
PandasTools.AddMoleculeColumnToFrame(output_df, smilesCol="smiles")

# Sort molecules by pIC50
output_df.sort_values(by="pIC50", ascending=False, inplace=True)

# Reset index
output_df.reset_index(drop=True, inplace=True)

output_df.drop("smiles", axis=1).head(3)

output_df

"""**Save Dataset for Next Session**"""

output_df.to_csv("/content/NS3.csv")
output_df.head()

print(f"DataFrame shape: {output_df.shape}")
# NBVAL_CHECK_OUTPUT
