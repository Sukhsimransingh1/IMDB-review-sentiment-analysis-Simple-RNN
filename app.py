import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the imdb dataset
word_index=imdb.get_word_index()
reverse_word_index={ value:key for key,value in word_index.items()}

#Load the pre trained model
model=load_model('simple.rnn_imdb.h5')

#decode function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+ 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment,prediction[0][0]

#streamlit app
import streamlit as st
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.markdown("# 🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Analyze whether a movie review expresses **positive** or **negative** sentiment using a Simple RNN model.")

st.divider()

st.subheader("✍️ Enter your movie review")
user_input = st.text_area(
    "Type your review here...",
    height=150,
    placeholder="Example: The movie was fantastic with great acting and story."
)

col1, col2 = st.columns([1, 2])

with col1:
    classify_btn = st.button("🔍 Classify Review")

with col2:
    st.caption("Model: Simple RNN | Dataset: IMDB")

st.divider()

if classify_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review before classifying.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment, score = predict_sentiment(user_input)

        if sentiment == "Positive":
            st.success("😊 Sentiment: Positive")
        else:
            st.error("☹️ Sentiment: Negative")

        st.metric(label="Prediction Confidence", value=f"{score:.4f}")

st.markdown("---")
st.caption("Built with TensorFlow, Keras & Streamlit")