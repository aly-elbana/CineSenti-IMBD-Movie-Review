# 🎬 IMDB Movie Review — Sentiment Analysis

A compact, production-oriented repository for training and running an **LSTM-based sentiment classifier** on the **IMDB Movie Reviews** dataset.  
This project includes a single training notebook and a minimal **Streamlit inference app** for real-time predictions.

---

## 🚀 Key Features
- 🧠 **LSTM-based Deep Learning Model:** trained to classify reviews as *positive* or *negative* sentiment.  
- 📓 **Jupyter Notebook:** includes full pipeline — data loading, preprocessing, training, and model export.  
- 💻 **Streamlit App:** simple web interface for local inference and testing.  
- 🔁 **Reproducible Setup:** minimal, well-documented, and easy to run end-to-end.  
- 🗂️ **Lightweight Repo:** large files (models/datasets) excluded with `.gitignore` for cleaner version control.

---

## 📁 Repository Structure

```
imdb-movie-review/
├── imbd_movie_review.ipynb      # Main training notebook
├── app.py                       # Streamlit app for inference
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files (models, datasets, etc.)
└── README.md                    # Project documentation
```

If you save trained artifacts, they should be placed **alongside the project root** (ignored by `.gitignore`):

```
models/ (gitignored)
├── sent-analysis.keras          # Saved Keras model
└── tokenizer.pickle             # Saved tokenizer for inference
```

---

## ⚡ Quickstart (Windows)

### 1️⃣ Clone and enter the project
```bash
git clone <your-repo-url>
cd imdb-movie-review
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model (optional)
Open the notebook in **Jupyter** or **VS Code** and run the cells from top to bottom.

After training, save your model and tokenizer so that the Streamlit app can load them:
```python
import pickle

# Save tokenizer
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

# Save trained model
model.save("sent-analysis.keras")
```

---

### 5️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`) to access the web UI.  
Paste a movie review and click **Predict** to get instant sentiment analysis.

---

## 🧩 Notes & Recommendations
- The notebook uses **kagglehub** to fetch both:
  - `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
  - `danielwillgeorge/glove6b100dtxt` (GloVe embeddings)
- For offline runs, download the dataset manually and place them inside a `data/` directory:
  ```
  data/
  ├── IMDB Dataset.csv
  └── glove.6B.100d.txt
  ```
  Then update the notebook paths accordingly.
- Ensure `MAXLEN` in `app.py` matches the sequence length used in training (`MAXLEN = 100`).
- Large files (models, raw data) are excluded from Git via `.gitignore` to keep the repository clean.

---

## ❤️ Future Enhancements (Optional)
If you’d like, I can help you:
- Add a **notebook cell** that automatically saves the model and tokenizer after training (copy-paste ready).  
- Replace **kagglehub** steps with local `data/` loading instructions for offline environments.

---

