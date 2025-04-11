# 🎬 Movie Recommendation System

This project recommends similar movies based on a selected title using NLP and cosine similarity.

## 📌 Features

- Uses metadata: genres, cast, keywords, crew, overview
- Processes with NLTK + Scikit-learn
- Computes cosine similarity between movies
- Fast recommendations without needing deep learning!

## 🛠️ Stack

- Python
- Pandas, scikit-learn, NLTK
- Pickle (for storing models)
- TMDB 5000 dataset

## 📂 Dataset

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## 🧪 How to Run

```bash
pip install -r requirements.txt
python recommend.py
