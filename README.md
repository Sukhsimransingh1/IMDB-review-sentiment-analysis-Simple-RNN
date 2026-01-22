# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

This project implements a Sentiment Analysis system for IMDB movie reviews using a Simple Recurrent Neural Network (RNN).
The model classifies a given movie review as Positive or Negative based on the textual content.
# Project Overview
Dataset: IMDB Movie Reviews (Keras built-in dataset)
Task: Binary text classification (Positive / Negative)
Model: Simple RNN-based Neural Network
Framework: TensorFlow / Keras
The project demonstrates how recurrent neural networks can be applied to natural language processing tasks such as sentiment analysis.
# ⚙️ Workflow
Load and preprocess the IMDB dataset
Perform text tokenization and sequence padding
Build a Simple RNN model with embedding and dense layers
Train the model on labeled movie reviews
Evaluate performance on test data
Predict sentiment for new user reviews
# 🧠 Model Architecture
Embedding Layer
Simple RNN Layer
Dense Output layer with Sigmoid Activation
# 📊 Results
The model is able to classify short and medium-length reviews with reasonable accuracy and correctly identifies basic positive and negative sentiments.
# ⚠️ Limitation
Simple RNN struggles with long reviews because it cannot effectively remember long-term context. As a result, performance may decrease for lengthy and complex sentences.
# Technologies Used
Python
TensorFlow / Keras
NumPy
Jupyter Notebook
