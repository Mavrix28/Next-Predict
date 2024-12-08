import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('hamlet_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if(len(token_list) >= max_sequence_len):
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Define the predict_next_word function
def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    # Tokenize input text
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    
    # Pad the sequence
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')

    # Predict the next word
    predictions = model.predict(padded_sequence)
    predicted_word_index = np.argmax(predictions)
    
    # Retrieve the word from tokenizer's index
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"


# Streamlit App
st.title("Next Word Prediction with LSTM Model")
st.write("This app uses an LSTM model to predict the next word in a sentence. Enter a sentence and the app will predict the next word.")





input_text = st.text_input("Enter text:")

# Placeholder for prediction message
prediction_placeholder = st.empty()

if st.button("Predict"):
    if not input_text.strip():  # Check if the input text is empty
        prediction_placeholder.write("Please enter a sentence.")
    else:
        try:
            # Validate the model's input shape and retrieve max sequence length
            max_sequence_len = model.input_shape[1]  # Sequence length expected by the model

            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            prediction_placeholder.write(f"The predicted next word is: {next_word}")
        except Exception as e:
            # Handle errors
            prediction_placeholder.write(f"An error occurred: {str(e)}")



