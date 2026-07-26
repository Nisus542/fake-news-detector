# Suggested EDA for your report

You don't need a full Jupyter notebook, but running these and screenshotting
the output/charts will strengthen the "Data Science" section of your report.

## 1. Class balance
Already saved automatically as `models/class_balance.png` after training.

## 2. Article length distribution
```python
import pandas as pd
df = pd.read_csv("data/True.csv")
df["text"].str.split().str.len().describe()
```
Compare fake vs. real — fake articles are often shorter or more erratic in length.

## 3. Most common words per class
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=20)
counts = cv.fit_transform(df["text"])
list(zip(cv.get_feature_names_out(), counts.sum(axis=0).tolist()[0]))
```
Run separately on Fake.csv and True.csv, compare top words — often fake news
skews toward more emotionally charged / sensational vocabulary.

## 4. Punctuation / capitalization signals
```python
df["exclaim_count"] = df["text"].str.count("!")
df["caps_ratio"] = df["text"].apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
```
Compare averages between fake and real — a nice, simple, explainable feature
to mention even if you don't add it to the model.

## What to put in the report
- A short paragraph + the class balance chart
- A short paragraph + article length comparison
- A short table of top words per class
- Tie it back to your model: "These patterns motivated using TF-IDF with
  bigrams, since word choice and phrasing patterns are strong signals here."
