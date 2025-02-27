import streamlit as st
import requests

# Flask endpoint for predictions
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    if user_input:
        try:
            # Sending the request to Flask backend for prediction
            response = requests.post(prediction_endpoint, json={"text": user_input})
            response.raise_for_status()  # Check for errors in the request
            
            # Handling the response from Flask
            response = response.json()
            if "prediction" in response:
                st.write(f"Predicted sentiment: {response['prediction']}")
            else:
                st.write("Error:", response.get("error", "Unknown error occurred."))
        except requests.exceptions.RequestException as e:
            st.write(f"Error: {e}")
    else:
        st.write("Please enter some text to predict.")