# 🎬 Movie Recommendation System (Content-Based)

This project is a **content-based movie recommendation system** that suggests similar movies based on textual metadata like cast, genre, director, keywords, and plot overview. It uses **classical NLP techniques** to compute movie similarity and generate recommendations.

---

## 🚀 Features

- Content-based recommendations
- Uses metadata from TMDB 5000 dataset
- Text preprocessing using NLP
- Feature engineering with Bag-of-Words model
- Cosine similarity for recommendation scoring
- Fast and interpretable

---

## 📁 Dataset Used

The system uses the **TMDB 5000 Movies & Credits dataset**, which includes:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These datasets contain metadata like:
- Movie titles
- Overviews
- Cast and crew
- Genres
- Keywords
- Ratings and popularity

---

## 🧠 Techniques Used

### 1. **Feature Extraction**
From each movie, we extract:
- **Genres**
- **Cast**
- **Crew** (Director only)
- **Keywords**
- **Overview**

These are processed and **merged into a single string field** (`tags`) per movie, which acts as a content "signature" of the film.

---

### 2. **Text Preprocessing**
- Lowercasing
- Removing spaces
- Tokenization
- Stemming using `PorterStemmer` (from `nltk.stem.porter`)

> Example: `"Running"` → `"run"`

This step ensures consistent token representation.

---

### 3. **Vectorization: Count Vectorizer (Bag-of-Words Model)**

We convert the `tags` column into numerical vectors using **CountVectorizer** from `scikit-learn`.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
```

- Each movie is represented as a **5000-dimensional sparse vector**.
- Each dimension corresponds to the frequency of a token.
- This is a classical NLP technique known as **Bag-of-Words (BoW)**.

---

### 4. **Similarity Calculation: Cosine Similarity**

We calculate **cosine similarity** between all movie vectors to get a movie-to-movie similarity matrix.

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

- Values range from 0 (no similarity) to 1 (perfect similarity).
- The matrix is used to find top-k similar movies.

---

### 5. **Recommendation Function**

The core recommendation logic:
- Input: Movie title
- Find the index of the movie in the dataset
- Fetch similarity scores from the matrix
- Sort and return the top 5 most similar movies

```python
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    sorted_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in sorted_movies:
        print(movies.iloc[i[0]].title)
```

---

##  Output Example

```python
recommend("Batman Begins")
```

🎩 Output:
```
The Dark Knight
Batman
Batman v Superman: Dawn of Justice
The Dark Knight Rises
The Prestige
```

---


## 🔧 Requirements

Install packages using:

```bash
pip install pandas numpy scikit-learn nltk
```

---

## 🧠 Summary of Techniques

| Component            | Technique                      |
|---------------------|--------------------------------|
| Feature Extraction  | NLP preprocessing              |
| Text Cleaning       | Tokenization + Stemming        |
| Vectorization       | `CountVectorizer` (BoW model)  |
| Similarity Measure  | Cosine Similarity              |
| ML Libraries Used   | scikit-learn, NLTK             |
| Data Format         | Pickle for storing models      |

---

## 🙌 Credits

- Dataset from [Kaggle TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- NLP and ML techniques inspired by classical recommendation systems

---
