# 🧠 News Embeddings and Impact Prediction

This repository contains a full machine learning pipeline for predicting the **impact of news articles on stock prices**, based on labeled financial news. The system processes raw news data, generates embeddings using transformer models, trains and evaluates multiple regression models, and supports fine-tuning and prediction with proper weighting.

---

## 📌 Project Overview

The project is divided into the following stages:

1. **Generating Embeddings**
      - 📄 *Notebook: `transformers_embeddings.ipynb`*
      - A pre-trained transformer model from Hugging Face is used.
      - The **CLS token embedding** of each news article is extracted and used as its final representation.
2. **Training the Model and Saving Weights**
   - 📄 *Notebook: `regression_models_training.ipynb`*
3. **Predicting impact with use of trained models**
   - 📄 *Notebook: `final_pipeline.ipynb`*

---

## 📁 Project Structure
<pre> 
  .
  ├── custom_modules/ 
  │ ├── __init__.py 
  │ ├── embeddinger.py                     # Embedding logic with transformer model 
  │ ├── parameters_search.py               # Grid/Random search for hyperparameters 
  │ └── predict.py                         # Inference pipeline 
  ├── data/ 
  │ ├── concatenated_news.xlsx 
  │ ├── df_embs_tiny.xlsx 
  │ ├── generated_and_bad_news.xlsx 
  │ └── knn_svr_real_test_1.xlsx 
  ├── pkl_models/ 
  │ ├── StackingReg.pkl 
  │ ├── regress_knn.pkl 
  │ └── regress_svr.pkl 
  ├── transformers_embeddings.ipynb        # Generate embeddings 
  ├── regression_models_training.ipynb     # Train and evaluate regressors 
  ├── final_pipeline.ipynb                 # Load model and predict 
 </pre>

---

## 🚀 Features
Transformer-based semantic embeddings

Sample weighting for synthetic data

Hyperparameter search and model stacking

Modular pipeline with reusable scripts

---

## 📈 Use Cases
Predicting stock price movements from news

Event-based financial modeling

Generating semantic embeddings for finance

Backtesting news impact strategies

---

## 📝 Author

**Artur Garipov**  
[LinkedIn](https://www.linkedin.com/in/artur-garipov-36037a319) | [GitHub](https://github.com/Artur-Gar)
