# IMDB Movie Review — Sentiment Analysis

A compact, production-oriented repository for training and running an LSTM-based sentiment classifier on the IMDB movie reviews dataset. The repository includes a single training notebook and a minimal Streamlit inference app.

Key points

- Minimal, well-documented codebase focused on reproducibility.
- Notebook contains data loading, preprocessing, training, and model export.
- Streamlit app provides a simple interface for local inference.

Repository tree

- Shows every file in the project root and recommended artifacts.

imdb-movie-review/
├── imbd_movie_review.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

If you save trained artifacts they should be placed alongside the project root (and are ignored by .gitignore):
models/ (gitignored)
├── sent-analysis.keras # saved Keras model (binary)
└── tokenizer.pickle # saved tokenizer for inference

Quickstart (Windows)

1. Clone and enter project
   git clone <your-repo-url>
   cd imbd-movie-review

2. Create and activate a virtual environment
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Train (optional)

   - Open `imbd_movie_review.ipynb` in Jupyter or VS Code and run the cells from top to bottom.
   - After training, save the tokenizer and model so the Streamlit app can load them:

     ```python
     import pickle
     with open("tokenizer.pickle", "wb") as f:
         pickle.dump(tokenizer, f)

     model.save("sent-analysis.keras")
     ```

5. Run the Streamlit app
   streamlit run app.py
   - Follow the local URL printed by Streamlit to open the UI in your browser.
   - Paste a review and click Predict.

Notes

- The notebook currently uses kagglehub to fetch the IMDB CSV and GloVe embeddings. For a local run, download the dataset and glove file to a `data/` folder and update the notebook paths.
- Ensure MAXLEN in `app.py` matches the max sequence length used during training (notebook uses MAXLEN=100).
- Large files (models, raw data) are excluded via `.gitignore` to keep the repository lightweight.

If you want, I can:

- Add the notebook cell that auto-saves tokenizer/model after training (copy/paste ready), or
- Replace the kagglehub steps with local `data/IMDB Dataset.csv` instructions.
