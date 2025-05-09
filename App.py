import streamlit as st
import joblib
import numpy as np

# Load your trained model (ensure you have a model.pkl file saved after training)
model = joblib.load("fake_news_model.pkl")  # Replace with your actual model path

# Function to make predictions
def predict_news(news_text):
    # Preprocess the input text and convert it to a format suitable for the model (e.g., TF-IDF vectorization)
    # Assuming the model uses a vectorizer like TfidfVectorizer
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure you have the vectorizer saved
    news_text_vectorized = vectorizer.transform([news_text])

    # Predict if the news is fake or real
    prediction = model.predict(news_text_vectorized)
    
    # Return the prediction (1 = Fake, 0 = Real)
    return "Fake News" if prediction == 1 else "Real News"

# Streamlit App Layout
st.title('Fake News Detection')
st.markdown('Enter the news text below to check if it is real or fake:')

# Text input from user
news_input = st.text_area("Enter News Text:")

# Prediction button
if st.button("Predict"):
    if news_input:
        result = predict_news(news_input)
        st.write(f"The news is: {result}")
    else:
        st.write("Please enter some text to analyze.")

