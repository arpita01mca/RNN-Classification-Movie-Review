# main.py

## Step 1: Import libraries
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Step 2: Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 3: Load the pretrained model safely
# Use compile=False to avoid legacy optimizer/loss issues with Keras 3
model = load_model('SimpleRNN/simple_rnn_imdb.h5', compile=False)

# Step 4: Helper Functions

def decoded_review(encoded_review):
    """
    Convert encoded review back to words.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text, maxlen=500):
    """
    Preprocess user input text to match model input format.
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown word
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

# Step 5: Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Step 6: Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

# Button to classify
if st.button('Classify'):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {score:.4f}')
    else:
        st.write('Please enter a movie review.')
