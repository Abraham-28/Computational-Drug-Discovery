# -*- coding: utf-8 -*-
"""CDD_Part_3(Removal_of_Unwanted_Compounds).ipynb

# Refer my google colab link for notebook file https://colab.research.google.com/drive/1vM9h9CTdejIefpsoM34D35c0Qtr-_c2x#scrollTo=Nb5zt_FjH_mU
# **Removal of the Unwanted Compounds From the last Dataset**

## **Basic Concepts**

There are some substructures we prefer not to include into our screening library. Before going to the implementation, let's see the characteristics of such unwanted substructures.

**Unwanted substructures**

Substructures can be unfavorable, e.g., because they are toxic or reactive, due to unfavorable pharmacokinetic properties, or because they likely interfere with certain assays.
Nowadays, drug discovery campaigns often involve [high throughput screening](https://en.wikipedia.org/wiki/High-throughput_screening). Filtering unwanted substructures can support assembling more efficient screening libraries, which can save time and resources.

Brenk *et al.* ([_Chem. Med. Chem._ (2008), **3**, 435-44](https://onlinelibrary.wiley.com/doi/full/10.1002/cmdc.200700139)) have assembled a list of unfavorable substructures to filter their libraries used to screen for compounds to treat neglected diseases. Examples of such unwanted features are nitro groups (mutagenic), sulfates and phosphates (likely resulting in unfavorable pharmacokinetic properties), 2-halopyridines and thiols (reactive).

**Pan Assay Interference Compounds (PAINS)**

[PAINS](https://en.wikipedia.org/wiki/Pan-assay_interference_compounds) are compounds that often occur as hits in HTS even though they actually are false positives. PAINS show activity at numerous targets rather than one specific target. Such behavior results from unspecific binding or interaction with assay components. Baell *et al.* ([_J. Med. Chem._ (2010), **53**, 2719-2740](https://pubs.acs.org/doi/abs/10.1021/jm901137j)) focused on substructures interfering in assay signaling. They described substructures which can help to identify such PAINS and provided a list which can be used for substructure filtering.

In RDKit, PAINS (Pan Assay Interference Compounds) refers to a list of substructures that are known to cause false positives in drug discovery assays. RDKit provides tools to filter out molecules containing these unwanted substructures within the “FilterCatalog” module.

![PAINS](https://github.com/volkamerlab/teachopencadd/blob/master/teachopencadd/talktorials/T003_compound_unwanted_substructures/images/PAINS_Figure.jpeg?raw=1)

Figure 1: Specific and unspecific binding in the context of PAINS. Figure taken from [Wikipedia](https://commons.wikimedia.org/wiki/File:PAINS_Figure.tif).

### Contents

* Load and visualize data
* Filter for PAINS
* Filter for unwanted substructures
* Highlight substructures
* Substructure statistics

### References

* Pan Assay Interference compounds ([wikipedia](https://en.wikipedia.org/wiki/Pan-assay_interference_compounds), [_J. Med. Chem._ (2010), **53**, 2719-2740](https://pubs.acs.org/doi/abs/10.1021/jm901137j))
* Unwanted substructures according to Brenk *et al.* ([_Chem. Med. Chem._ (2008), **3**, 435-44](https://onlinelibrary.wiley.com/doi/full/10.1002/cmdc.200700139))
* Inspired by a Teach-Discover-Treat tutorial ([repository](https://github.com/sriniker/TDT-tutorial-2014/blob/master/TDT_challenge_tutorial.ipynb))
* RDKit ([repository](https://github.com/rdkit/rdkit), [documentation](https://www.rdkit.org/docs/index.html))

## Practical

### Load and visualize data

###Installation and Imports
"""

! pip install rdkit

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Load the data generated in Part-2
NS3_data = pd.read_csv('/content/2-ro5_properties_fulfilled.csv',
    index_col=0,
)
# Drop unnecessary information
print("Dataframe shape:", NS3_data.shape)
NS3_data.drop(columns=["molecular_weight", "n_hbd", "n_hba", "logp"], inplace=True)
NS3_data.head()

# Add molecule column
PandasTools.AddMoleculeColumnToFrame(NS3_data, smilesCol="smiles")
# Draw first 5 molecules
Chem.Draw.MolsToGridImage(
    list(NS3_data.head(5).ROMol),
    legends=list(NS3_data.head(5).molecule_chembl_id),
)

"""### Filter for PAINS

The PAINS filter is implemented in RDKit ([documentation](http://rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html)).
"""

# initialize filter
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)

# search for PAINS
matches = []
clean = []
for index, row in tqdm(NS3_data.iterrows(), total=NS3_data.shape[0]):
    molecule = Chem.MolFromSmiles(row.smiles)
    entry = catalog.GetFirstMatch(molecule)  # Get the first matching PAINS
    if entry is not None:
        # store PAINS information
        matches.append(
            {
                "chembl_id": row.molecule_chembl_id,
                "rdkit_molecule": molecule,
                "pains": entry.GetDescription().capitalize(),
            }
        )
    else:
        # collect indices of molecules without PAINS
        clean.append(index)

matches = pd.DataFrame(matches)
NS3_data = NS3_data.loc[clean]  # keep molecules without PAINS

# NBVAL_CHECK_OUTPUT
print(f"Number of compounds with PAINS: {len(matches)}")
print(f"Number of compounds without PAINS: {len(NS3_data)}")

#Let's have a look at the first 5 identified PAINS.

Chem.Draw.MolsToGridImage(
    list(matches.head(5).rdkit_molecule),
    legends=list(matches.head(5)["pains"]),
)

"""### Filter and highlight unwanted substructures

Some lists of unwanted substructures, like PAINS, are already implemented in RDKit. And let's do a further filtering using external list and get the substructure matches manually.

Here, we use the list provided in the supporting information from Brenk *et al.* ([_Chem. Med. Chem._ (2008), **3**, 535-44](https://onlinelibrary.wiley.com/doi/full/10.1002/cmdc.200700139)).
"""

substructures = pd.read_csv("/content/3_unwanted_substructures.csv", sep="\s+")
substructures["rdkit_molecule"] = substructures.smarts.apply(Chem.MolFromSmarts)
print("Number of unwanted substructures in collection:", len(substructures))
# NBVAL_CHECK_OUTPUT

#Let's have a look at a few of these substructures.

Chem.Draw.MolsToGridImage(
    mols=substructures.rdkit_molecule.tolist()[2:5],
    legends=substructures.name.tolist()[2:5],
)

"""Search our filtered dataframe for matches with these unwanted substructures."""

# search for unwanted substructure from our dataset
matches = []
clean = []
for index, row in tqdm(NS3_data.iterrows(), total= NS3_data.shape[0]):
    molecule = Chem.MolFromSmiles(row.smiles)
    match = False
    for _, substructure in substructures.iterrows():
        if molecule.HasSubstructMatch(substructure.rdkit_molecule):
            matches.append(
                {
                    "chembl_id": row.molecule_chembl_id,
                    "rdkit_molecule": molecule,
                    "substructure": substructure.rdkit_molecule,
                    "substructure_name": substructure["name"],
                }
            )
            match = True
    if not match:
        clean.append(index)

matches = pd.DataFrame(matches)
NS3_data = NS3_data.loc[clean]

# NBVAL_CHECK_OUTPUT
print(f"Number of found unwanted substructure: {len(matches)}")
print(f"Number of compounds without unwanted substructure: {len(NS3_data)}")

"""### Highlight substructures

Let's have a look at the first 5 identified unwanted substructures. Since we have access to the underlying SMARTS patterns we can highlight the substructures within the RDKit molecules.
"""

#Let's look at the first 5 identified unwanted substructures.

to_highlight = [
    row.rdkit_molecule.GetSubstructMatch(row.substructure) for _, row in matches.head(5).iterrows()
]
Chem.Draw.MolsToGridImage(
    list(matches.head(5).rdkit_molecule),
    highlightAtomLists=to_highlight,
    legends=list(matches.head(5).substructure_name),
)

"""### Substructure statistics

Now let's find out the most frequent substructure found in our data set. The Pandas `DataFrame` provides convenient methods to group containing data and to retrieve group sizes.
"""

# NBVAL_CHECK_OUTPUT
groups = matches.groupby("substructure_name")
group_frequencies = groups.size()
group_frequencies.sort_values(ascending=False, inplace=True)
group_frequencies.head(10)

"""**Save the datasets with and without unwanted compounds**"""

matches.to_csv("data_with_unwanted_compound.csv", index=True)

NS3_data.to_csv("Clean_NS3_data.csv", index=True)
