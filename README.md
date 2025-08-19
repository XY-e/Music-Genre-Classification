# Music Genre Classification (GTZAN Dataset)

This project implements **music genre classification** using the **GTZAN dataset**.  
We extract **MFCC features** from audio files using **Librosa** and train a **neural network** to classify songs into 10 genres.

---

## Project Description

The task is to classify audio tracks into one of **10 music genres**:  

`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`

### Workflow:
1. Download & preprocess dataset  
2. Extract audio features (MFCCs)  
3. Train a neural network model  
4. Evaluate model performance (accuracy, loss, plots)  

---

## Dataset

- **Dataset:** [GTZAN - Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- **Classes:** 10  
- **Samples:** 1000 audio files (30s each, 22050Hz)  

---

## Feature Extraction
- We use MFCCs (Mel-Frequency Cepstral Coefficients) as features.
- Each audio file is converted into a 40-dimensional MFCC vector (averaged over time).

---

## Model Architecture
- A simple Fully Connected Neural Network (MLP) was trained:
  - Dense(256, ReLU) + Dropout(0.3)
  - Dense(128, ReLU) + Dropout(0.3)
  - Dense(10, Softmax)
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

---

## Training & Results
- Training History (example run)
- Epochs: 30
- Training Accuracy: ~86%
- Validation Accuracy: ~61%
- Final Test Accuracy: ~60%

---

## Setup & Installation

```bash
# Install dependencies
pip install kaggle kagglehub librosa pandas matplotlib seaborn scikit-learn tensorflow
