import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

st.title("📰 Stock News Sentiment Analysis")
user_input = st.text_area("Enter stock-related news headline:")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    label = "📈 Positive" if prediction == 1 else "📉 Negative"
    st.success(f"Prediction: {label}")
