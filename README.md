# Tackling Protein Function Prediction Using Deep Learning Techniques

This project implements & improves upon models found in O'Reilly's "Deep Learning for Biology".

## Highlights

## Deep-GO: Multi-Label Protein Function Prediction (`Proteins_my_models/`)

### Phase 1: Modeling Using Human-Only Proteins
* Predicting 1,900+ Gene Ontology terms from sequence alone using ESM-2 Embeddings and a Residual Label-Attention Architecture.
* **Architecture:** Residual Network + Label Attention Header, Purely Residual Network, and Standard (Fully Connected) Neural Network
* **Performance:** Achieved ~0.67 AUPRC and ~0.92 AUROC.
* **Embeddings:** Powered by ESM-2-640.

### Phase 2: All-Species Expansion (Currently In Progress)
* Expanded dataset to include all species, growing the label space to 6,000+ Gene Ontology terms
* Switched the protein language model (PLM) from ESM-2 to ESMC (600M parameter model) model- a newer PLM whose embedding outputs are of size 1,152. 
* Replaced the original random train/test split from sklearn with a homology-aware split:proteins are clustered by sequence identity using MMseqs2 (30% identity threshold, 80% coverage) so that no cluster of near-duplicate or homologous proteins is split across train/val/test — addressing a known source of data leakage and inflated benchmark scores in naive random splits for protein function prediction.
* **Architecture:** In progress- results to follow. 
* **Performance:** In progress- results to follow.
* **Embeddings:** In progress- results to follow. 



#### 🧬 The Scientific Problem
Current biological databases are exploding with protein sequences, but knowing what they actually do is slow and expensive as it requires inctricate experiments. This project automates that "annotation" by mapping high-dimensional protein language embeddings to the Gene Ontology (GO) hierarchy.

#### 🛠 Technical Innovation: Why this works
1. Label-Attention Mechanism:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A standard layer looks at all 1900+ labels as 
independent functions. In reality, many of these labels are related to one another ('metabolic regulation' and 'accelerating chemical reactions inside the cell'). To get the model to understand the sematic relationships between function labels, I used an attention mechanism in the `ProteinFunctionAttentionNetwork.py` model.

2. Handling Massive Class Imbalance:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To get the model to predict rare labels well, I utilized Focal Loss so easy, common examples are down-weighted and the model is forced to focus on the 
harder and rarer functional annotations.

3. Homology-Aware Data Splitting (Phase 2):\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Random train/test splits on protein sequence data risks leaking near-identical homologs across splits, letting a model "memorize" rather than generalize. I clustered all proteins by sequence identity with MMseqs2 and split at the cluster level, so evaluation reflects generalization to genuinly novel sequences.


#### 📊 Performance Visualization
Within `Proteins_my_models/ProteinFunctionNetwork_Results/`, `Proteins_my_models/ProteinFunctionResidualNetwork_Results/`, `Proteins_my_models/ProteinFunctionAttentionNetwork_Results/`, you'll find plots of these models at various 
hyperparameters.

#### 🚀 Future Roadmap
* Build & train new model on the all-species + ESMC-600M embeddings + ~6,192 GO label dataset.
* Compare Phase 1 (human-only) against Phase 2 (all-species) performance and analyze the difference that species expansion and embedding PLM choice affect model performance.



## Setup
1. Clone the repo.

2. Install dependencies: `pip install -r requirements.txt`

3. `pip install -e .` (installs the `Proteins_my_models` package so its internal imports resolve)

4. Run the data pipeline in order:
   1. Run through the cells of `protein_ML.ipynb` to gather data and perform basic EDA. Phase 1 models were prototyped in the Jupyter notebook as well. Phase 2 was just loading the data & initial EDA.
   2. `MMseqs2_Protein_Clustering.py` (Phase 2 Only) — clusters proteins by sequence identity and produces a homology-aware train/valid/test split
   3. `aa_embedding.py` (Phase 1 Only) / `aa_embedding_all.py` (Phase 2 Only) — generates PLM embeddings for each split
   4. `merge_aa_embedding.py` (Phase 1 Only) / `merge_all_species_aa_embedding.py` (Phase 2 Only) — consolidates embeddings and merges them with GO-term labels into training-ready files.
5. Train: run the training script in `Training_in_Docker/` (or your local equivalent) against the merged files (Available only for Phase 1 at the moment).