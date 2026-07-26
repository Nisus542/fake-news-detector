# TruthLens — Fake News Detector with GenAI Explanations

A Data Science & ML project with a GenAI layer on top, built for the Summer 2026
Extra Course Skill Development Cohort (Data Science & ML with GenAI).

## What it does

1. **ML layer (the "Data Science / ML" half):** A TF-IDF + Logistic Regression
   classifier trained on the Kaggle Fake and Real News Dataset predicts whether
   a pasted article is likely real or fake, with a confidence score.
2. **GenAI layer (the "GenAI" half):** Once the ML model makes its prediction,
   an LLM (via Groq) explains *why* the article looks suspicious or credible —
   flagging emotionally loaded language, unverified claims, and suggesting
   what to fact-check. The LLM explains the model's reasoning; it does not
   make the real/fake call itself — that separation is the whole point of the
   project and is worth calling out explicitly in the report.

## Results

Trained on the full Kaggle dataset (44,889 articles):
- Accuracy: 98.9%
- Precision: 98.7%
- Recall: 99.0%
- F1: 98.9%

Note: this dataset has fairly distinct structural differences between its
real and fake articles (e.g. formatting conventions), which makes it easier
to separate than real-world "in the wild" news would be. That's part of why
it's a common teaching dataset — but it's worth stating plainly in the report
that ~99% reflects strong signal in this dataset, not a guarantee of the same
performance on live, unseen articles.

## Project structure

```
fake-news-detector/
├── data/
│   └── README.md
├── models/
├── notebooks/
│   └── exploration.md
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── generate_sample_data.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup

```powershell
pip install -r requirements.txt
```

## Step 1 — Get the dataset

Download the **Fake and Real News Dataset** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Put `Fake.csv` and `True.csv` in the `data/` folder. These files are not
committed to this repo (too large for GitHub) — see `data/README.md`.

## Step 2 — Train the model

```powershell
python src/train_model.py
```

This loads and cleans the data, trains the classifier, prints evaluation
metrics, and saves `model.joblib`, `vectorizer.joblib`, and evaluation charts
(`confusion_matrix.png`, `metrics.png`, `class_balance.png`) to `models/`.

## Step 3 — Set up your Groq API key

Sign up free at https://console.groq.com and create an API key.

Create a `.env` file in the project root:

GROQ_API_KEY=your_key_here

`.env` is already excluded via `.gitignore` — never commit it.

## Step 4 — Run the app

```powershell
streamlit run app.py
```

Paste any news article text and get a Real/Fake prediction with confidence,
plus a GenAI-generated explanation of the reasoning.

## Deploying for the live demo link

Repo is pushed to GitHub: https://github.com/Nisus542/fake-news-detector

To deploy: connect the repo on Streamlit Community Cloud
(share.streamlit.io), and add `GROQ_API_KEY` as a secret in the app's
settings (Streamlit Cloud uses its own secrets manager, not `.env`).

## For the report

- `notebooks/exploration.md` has suggested EDA angles (class balance, article
  length distributions, common words per class)
- `models/confusion_matrix.png`, `metrics.png`, and `class_balance.png` are
  ready to drop directly into the report
- Worth naming as a limitation: the dataset is a few years old and skews
  toward US political news, so it may generalize less well to very recent or
  non-political topics