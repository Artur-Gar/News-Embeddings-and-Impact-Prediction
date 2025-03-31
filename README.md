# 🧠 News Embeddings and Impact Prediction

This repository contains a full machine learning pipeline for predicting the **impact of news articles on stock prices**, based on labeled financial news. The system processes raw news data, generates embeddings using transformer models, trains and evaluates multiple regression models, and supports fine-tuning and prediction with proper weighting.

---

## 📌 Problem Statement

"""
Based on the available news and their impact labels, the task is to train an ML model to predict the impact of a news article on a company's stock price.
"""

---

## 🔄 Project Pipeline

The project is divided into the following stages:

1. **Generating Embeddings**
2. **Training the Model and Saving Weights**

---

## 🔎 Stage 1: Generating Embeddings  
📄 *Notebook: `transformers_embeddings.ipynb`*

- A pre-trained transformer model from Hugging Face is used.
- Since internet access was restricted, the model was downloaded and loaded via **local paths**.
- The **CLS token embedding** of each news article is extracted and used as its final representation.

---

## 🤖 Stage 2: Model Training and Weight Saving  
📄 *Notebook: `regression_models_training.ipynb`*

- **Penalty weights** (e.g., `0.75`) are assigned to synthetic or second-round generated news to avoid overfitting:
  ```python
  model.fit(X_train, y_train, sample_weight=sample_weight)
