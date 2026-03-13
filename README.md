# Twitter Sentiment Analysis

A machine learning project that classifies tweets as positive or negative using logistic regression and TF-IDF text features. Trained on 1.6 million tweets from the Sentiment140 dataset.

---

## What it does

You feed it a tweet, it tells you whether the sentiment is positive or negative. That's it. The model runs fast, stays interpretable, and hits ~77-78% accuracy on held-out data — which is a reasonable baseline for noisy social media text without using any deep learning.

---

## Dataset

The project uses the [Sentiment140 dataset](http://help.sentiment140.com/for-students) — 1.6 million tweets collected in 2009, pre-labeled using emoticons as a proxy for sentiment. Positive tweets (originally labeled `4`) are mapped to `1`, and negative tweets (labeled `0`) stay as `0`. The dataset is perfectly balanced: 800,000 of each class, so no resampling was needed.

The CSV has six columns — `target`, `id`, `date`, `flag`, `user`, and `text` — though only `target` and `text` are used for training.

---

## How it works

**Step 1 — Text cleaning and stemming.** Raw tweets are messy: @mentions, URLs, punctuation, and stopwords like "the" or "is" all add noise without adding meaning. A custom `stemming()` function strips non-alphabetic characters, lowercases everything, drops English stopwords (via NLTK), and reduces each remaining word to its root form using Porter Stemmer. So "acting", "actress", and "actor" all become "act". This shrinks the vocabulary and makes the model more generalizable.

**Step 2 — TF-IDF vectorization.** The cleaned text gets converted to numbers using `TfidfVectorizer`. TF-IDF scores each word by how often it appears in a tweet relative to how common it is across all tweets. Words that show up in every tweet get low weight; words that are distinctive to a handful of tweets get higher weight. The resulting matrix has 1.28 million rows and ~461,000 columns — one column per unique token in the vocabulary.

**Step 3 — Logistic regression.** The vectorized tweets go into a logistic regression classifier. It draws a decision boundary in that high-dimensional word space and learns which word patterns lean positive versus negative. Simple, but effective.

**Step 4 — Train/test split.** The data is split 80/20 with `stratify=Y`, meaning both splits maintain the 50/50 class balance. This gives 1,280,000 training examples and 320,000 for testing.

---

## Results

| Split | Accuracy |
|-------|----------|
| Training | ~79.9% |
| Test | ~77.7% |

The small gap between training and test accuracy means the model is not overfitting — it generalizes reasonably well to tweets it has never seen.

---

## Project structure

```
├── sentimentanaly.ipynb        # Main notebook
├── training.1600000.processed.noemoticon.csv   # Sentiment140 dataset (not included)
├── trained_model.pkl           # Saved logistic regression model
└── README.md
```

---

## Requirements

```
numpy
pandas
scikit-learn
nltk
```

Install with:

```bash
pip install numpy pandas scikit-learn nltk
```

You also need NLTK's stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

---

## Running the notebook

Download the Sentiment140 dataset and place the CSV in the same directory as the notebook. The file should be named `training.1600000.processed.noemoticon.csv`. Then run all cells in order. The final cells save the trained model to `trained_model.pkl` and run a quick sanity check prediction.

---

## Making predictions on new data

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

# Preprocess your tweet the same way as training data (stemming + vectorizing),
# then predict
prediction = loaded_model.predict(X_new)

if prediction[0] == 0:
    print("Negative")
else:
    print("Positive")
```

Note: the input must go through the same stemming and TF-IDF transformation used during training. The vectorizer should be saved and loaded alongside the model to ensure consistent feature encoding.

---

## Limitations worth knowing

The model was trained on 2009 tweets, so it has no concept of slang that emerged after that. It also handles sarcasm poorly — a tweet like "oh great, another Monday" would likely score as positive because of the word "great". For production use, consider replacing the TF-IDF + logistic regression pipeline with a fine-tuned transformer like [BERTweet](https://huggingface.co/vinai/bertweet-base), which handles context and irony much better.

---

## License

This project is for educational purposes. The Sentiment140 dataset was created by Alec Go, Richa Bhayani, and Lei Huang at Stanford University.
