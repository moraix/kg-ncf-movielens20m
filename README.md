# 🎬 Enhancing Movie Recommendations with Knowledge Graph Embeddings and Neural Collaborative Filtering
This repository contains the full implementation of a hybrid recommender system that integrates Knowledge Graph Embeddings (KGE) with Neural Collaborative Filtering (NCF) to improve recommendation accuracy on sparse movie rating data.
The project combines structured information from IMDb with user-item interactions from MovieLens 20M, and evaluates the impact of KG-enhanced representations in a deep learning recommender.
## 📌 Project Overview
Recommender systems based solely on collaborative filtering suffer from sparsity and cold-start issues. This project addresses these challenges by incorporating a Knowledge Graph built from IMDb metadata (directors, actors, genres, writers) into a Neural Collaborative Filtering framework.
### Key Contributions
- Construction of a large-scale movie knowledge graph from IMDb
- Training TransE embeddings using PyKEEN
- High-coverage MovieLens–IMDb linkage
- Baseline NCF model for implicit feedback
- Hybrid KG-NCF with gated fusion
- Two-phase training for stable fusion learning
- Rigorous evaluation using top-K ranking metrics

## 🧩 Dataset Sources
- MovieLens 20M: User–item ratings (implicit feedback)
- IMDb Non-Commercial Datasets: Knowledge graph construction

⚠️ Raw datasets are not included in this repository due to licensing.

## 🏗️ System Pipeline
```
IMDb TSVs ──► Knowledge Graph ──► TransE (PyKEEN) ──► KGE
                                │
MovieLens ──► Linkage ───────────┘
    │
    └─► Implicit Ratings ──► Baseline NCF ──► Hybrid KG-NCF ──► Evaluation
```

## 🔬 Model Architecture
### Baseline: Neural Collaborative Filtering (NCF)

- User and item embeddings
- MLP interaction layers
- Implicit feedback training
- Negative sampling

### Hybrid KG-NCF

Item representation is enhanced using KG embeddings via gated fusion:
$$fused=g⋅ecf​+(1−g)⋅ekg$$
where:
$$g=σ(W[ecf​;ekg​])$$
### Two-Phase Training
- 1. Freeze CF layers and train KG fusion
- 2. Fine-tune the entire model
## 📊 Evaluation Results (Top-10)
| Model              | Precision@10 | Recall@10 | F1@10  | NDCG@10 |
| ------------------ | ------------ | --------- | ------ | ------- |
| Baseline NCF       | 0.3959       | 0.8673    | 0.4606 | 0.8361  |
| Hybrid KG-NCF      | 0.3886       | 0.8510    | 0.4520 | 0.8125  |
| Hybrid (Two-phase) | 0.3917       | 0.8596    | 0.4560 | 0.8228  |
Two-phase training significantly improves the stability and performance of KG fusion.
## 🗂️ Repository Structure
```
├── script.ipynb           # Full pipeline notebook
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
```
Large generated files (models, embeddings, filtered datasets) are excluded and generated during execution.
## ⚙️ Setup & Execution
### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Prepare datasets
Download:
- MovieLens 20M ratings
- IMDb Non-Commercial datasets
Place them in your working directory or Google Drive.
### 3. Run the pipeline
Execute script.ipynb step by step:
- 1. KG construction
- 2. TransE training
- 3. MovieLens–IMDb linkage
- 4. Baseline NCF
- 5. Hybrid KG-NCF
​
## 📁 Outputs

Generated artifacts include:
- Trained TransE embeddings
- NCF baseline and hybrid models
- Evaluation metrics
- Mapping files

These are saved locally or in Google Drive and are not tracked in Git.

## 🚀 Future Work

- Cold-start evaluation on low-interaction items
- Temporal modeling of user preferences
- Incorporation of textual reviews (multi-modal KG)
- Graph Neural Networks over KG
