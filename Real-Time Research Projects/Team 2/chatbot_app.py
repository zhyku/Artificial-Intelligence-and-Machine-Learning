import streamlit as st
import pandas as pd
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load DialoGPT ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# === Load Rick & Morty dataset ===
@st.cache_resource
def load_data():
    data = pd.read_csv("Rick and morty.csv")  # Make sure this file is in same directory
    data.dropna(subset=['name', 'line'], inplace=True)
    return data

data = load_data()

# === Preprocess text ===
def preprocess(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text.lower())
    return ' '.join(text.split())

# === Initialize TF-IDF ===
@st.cache_resource
def initialize_bot(data):
    lines = data['line'].apply(preprocess).tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lines)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = initialize_bot(data)

# === Respond using custom logic ===
def get_custom_response(user_input):
    user_input_processed = preprocess(user_input)
    input_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(input_vector, tfidf_matrix)
    best_match_idx = similarity.argmax()
    return data.iloc[best_match_idx]['name'] + ": " + data.iloc[best_match_idx]['line']

# === Fallback to DialoGPT ===
def get_dialo_gpt_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# === Streamlit App ===
st.set_page_config(page_title="Rick and Morty Chatbot", page_icon="🧪")
st.title("🧪 Chat with Rick and Morty!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Say something to Rick or Morty...")

if user_input:
    try:
        response = get_custom_response(user_input)
    except Exception as e:
        response = get_dialo_gpt_response(user_input)
    
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

for speaker, message in st.session_state.history:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(f"**{speaker}:** {message}")
