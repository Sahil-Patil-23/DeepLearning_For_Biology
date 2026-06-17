# Tackling Biological Problems Using Various Deep Learning Achitectures

This project implements & improves upon models found in O'Reilly's "Deep Learning for Biology".

## Highlights

## Deep-GO: Multi-Label Protein Function Prediction (Proteins_my_models)
* Predicting 1,900+ Gene Ontology terms from sequence alone using ESM-2 Embeddings and a Residual Label-Attention Architecture.
* **Architecture:** Residual Network + Label Attention Header, Purely Residual Network, and Standard (Fully Connected) Neural Network
* **Performance:** Achieved ~0.67 AUPRC and ~0.92 AUROC.
* **Embeddings:** Powered by ESM-2-640.
#### 🧬 The Scientific Problem
Current biological databases are exploding with protein sequences, but knowing what they actually do is slow and expensive as it requires inctricate experiments. This project automates that "annotation" by mapping high-dimensional protein language embeddings to the Gene Ontology (GO) hierarchy.

#### 🛠 Technical Innovation: Why this works
1. Label-Attention Mechanism:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A standard layer looks at all 1900+ labels as 
independent functions. In reality, many of these labels are related to one another ('metabolic regulation' and 'accelerating chemical reactions inside the cell'). To get the model to understand the sematic relationships between function labels, I used an attention mechanism in the `ProteinFunctionAttentionNetwork.py` model.

2. Handling Massive Class Imbalance:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To get the model to predict rare labels well, I utilized Focal Loss so easy, common examples are down-weighted and the model is forced to focus on the 
harder and rarer functional annotations.

#### 📊 Performance Visualization
Within `Proteins_my_models/ProteinFunctionNetwork_Results/`, `Proteins_my_models/ProteinFunctionResidualNetwork_Results/`, `Proteins_my_models/ProteinFunctionAttentionNetwork_Results/`, you'll find plots of these models at various 
hyperparameters.

#### 🚀 Future Roadmap


## Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
