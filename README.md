# Computational-Drug-Discovery
AI Driven Computational Drug Discovery using Graph Neural Networks for Dengue virus DENV-2 

ABSTRACT
Background: Dengue Virus Serotype 2 (DENV-2) poses a critical global health threat due to its severe complications such as dengue hemorrhagic fever and dengue shock syndrome. The absence of effective antiviral therapies and limitations in mosquito control strategies used currently is another concern. 
Objective: In this study we investigate the role of using Graph Neural Networks (GNNs) for computational drug discovery in order to identify potential antiviral drug candidates for DENV-2.
Methodology: We used our proposed hybrid (GCN&GAT) GNNs trained on the bioactivity dataset on DENV-2 from ChEMBL online database for computational drug discovery. LUNA22 ISMI dataset. We compared nine hybrid GNNs (GCN&GAT) models using three different loss functions (MSE, MAE &Huber loss) and three different augmented datasets. The baseline model we used for comparison with our proposed model is DGraphDTA model. (Jiang et al., 2020)
Result: From the nine hybrid (GCN&GAT) GNN models the best performing model is the model that uses MAE as a loss function and the last (clean NS3) augmented dataset for model training. The model achieved MSE of 0.0057 and R2 Score of 0.9938 accuracy, outperforming the baseline model for GNNs, which is DGraphDTA. Its performance on the Davis dataset was MSE of 0.202and on KIBA dataset MSE of 0.126.

Keywords: Dengue Virus, DENV-2, AI-Driven Drug Discovery, GNNs, VGAE, Molecular Docking, Drug Repurposing, Drug Designing, Computational Drug Discovery, Deep Learning, Healthcare, Viral Diseases.
