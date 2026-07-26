# TruthLens — Fake News Detector with GenAI Explanations

A Data Science & ML project with a GenAI layer on top, built for the Summer 2026
Extra Course Skill Development Cohort (Data Science & ML with GenAI).

## What it does

1. **ML layer (the "Data Science / ML" half):** A TF-IDF + Logistic Regression
   classifier trained on a labeled real/fake news dataset predicts whether a
   pasted article is likely real or fake, with a confidence score, plus
   accuracy/precision/recall/confusion-matrix visuals.
2. **GenAI layer (the "GenAI" half):** Once the ML model makes its prediction,
   an LLM (via Groq) explains *why* the article looks suspicious or credible —
   flagging emotionally loaded language, unverified claims, and suggesting
   what to fact-check. The LLM explains the model's reasoning; it does not
   make the real/fake call itself — that separation is the whole point of the
   project and is worth calling out explicitly in your report.

## Project structure

```
fake-news-detector/
├── data/
│   └── README.md          # how to get the dataset
├── models/                 # trained model + vectorizer land here after training
├── notebooks/
│   └── exploration.md      # suggested EDA steps for your report
├── src/
│   ├── preprocess.py       # text cleaning utilities
│   ├── train_model.py      # trains + evaluates the classifier, saves it
│   └── generate_sample_data.py  # makes a small synthetic dataset so you can
│                                  # test the whole pipeline before the real
│                                  # dataset is downloaded
├── app.py                  # Streamlit app (the demo / live app)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows (you're on Windows/ASUS per your setup)
pip install -r requirements.txt
```

## Step 1 — Get the dataset

Download the **Fake and Real News Dataset** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It gives you `Fake.csv` and `True.csv`. Put both files in the `data/` folder.

Don't have a Kaggle account handy or want to test the pipeline right now
without waiting? Run:

```bash
python src/generate_sample_data.py
```

This creates a small synthetic `data/Fake.csv` and `data/True.csv` so you can
run the entire pipeline end-to-end today. **Swap in the real Kaggle dataset
before your final training run** — the synthetic data is only for wiring
things up, accuracy numbers from it are meaningless for your report.

## Step 2 — Train the model

```bash
python src/train_model.py
```

This will:
- Load and clean the data
- Split into train/test
- Vectorize with TF-IDF
- Train a Logistic Regression classifier
- Print accuracy, precision, recall, F1, and a confusion matrix
- Save `models/model.joblib` and `models/vectorizer.joblib`
- Save evaluation charts to `models/` for use in your report

## Step 3 — Get a free Groq API key

Sign up at https://console.groq.com and create an API key (free tier is
generous and fast — this is the same service your Rosey chatbot uses).

Set it as an environment variable before running the app:

```bash
set GROQ_API_KEY=your_key_here        # Windows cmd
$env:GROQ_API_KEY="your_key_here"     # Windows PowerShell
```

## Step 4 — Run the app

```bash
streamlit run app.py
```

Paste or type any news article text into the box and get:
- A Real/Fake prediction with confidence score
- A plain-English GenAI explanation of the reasoning
- Highlighted suspicious phrases

## Deploying for your live demo link

Push this repo to GitHub, then deploy free on Streamlit Community Cloud
(share.streamlit.io) — connect your GitHub repo, add `GROQ_API_KEY` as a
secret in the app settings, and you'll get a public URL to put in your
submission form.

## For your report

- `notebooks/exploration.md` has suggested EDA angles (class balance, article
  length distributions, common words per class) worth including
- `models/` will contain your confusion matrix and metrics chart images after
  training — drop these straight into the report
- Worth naming as a limitation: the dataset is a few years old and skews
  toward US political news, so it may generalize less well to very recent or
  non-political topics
