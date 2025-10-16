import streamlit as st
import re
import string
import pickle
from bs4 import BeautifulSoup
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK resources (only downloads if missing)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

MAXLEN = 100  # must match notebook maxlen

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

st.title("IMDB Movie Review — Sentiment (LSTM + GloVe)")
st.markdown("Paste a movie review and get a prediction. Requires `sent-analysis.keras` and `tokenizer.pickle` in project root.")

# Load model + tokenizer
model = None
tokenizer = None
try:
    model = load_model("sent-analysis.keras")
except Exception as e:
    st.error("Model file `sent-analysis.keras` not found or failed to load. Train and save the model from the notebook.")
    st.stop()

try:
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
except Exception:
    st.error("Tokenizer file `tokenizer.pickle` not found. Save the tokenizer when training (see README).")
    st.stop()

review_input = st.text_area("Review", height=200)
if st.button("Predict") and review_input.strip():
    cleaned = preprocess_text(review_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, padding="post", maxlen=MAXLEN)
    pred = model.predict(seq, verbose=0)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    st.write(f"Prediction: **{label}**")
    st.write(f"Score: `{pred:.4f}` (threshold 0.5)")