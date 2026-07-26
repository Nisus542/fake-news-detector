"""
Trains a TF-IDF + Logistic Regression classifier on the Fake/Real news
dataset, evaluates it, and saves the model + vectorizer + evaluation charts.

Run from the project root:
    python src/train_model.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import clean_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    real_path = os.path.join(DATA_DIR, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(real_path)):
        raise FileNotFoundError(
            "Fake.csv / True.csv not found in data/. Either download the "
            "Kaggle dataset (see data/README.md) or run "
            "`python src/generate_sample_data.py` to create test data first."
        )

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 0  # 0 = fake
    real_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    df = df[df["content"].str.len() > 0]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Confusion matrix chart
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"]).plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics")
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "metrics.png"), dpi=150)
    plt.close()

    # Class balance chart (useful for report EDA section)
    fig, ax = plt.subplots(figsize=(5, 4))
    df["label"].map({0: "Fake", 1: "Real"}).value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class Balance")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "class_balance.png"), dpi=150)
    plt.close()

    return model, vectorizer, metrics


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} articles ({(df['label']==0).sum()} fake, "
          f"{(df['label']==1).sum()} real)")

    print("Training model...")
    model, vectorizer, metrics = train_and_evaluate(df)

    joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))

    print(f"\nSaved model + vectorizer to {MODELS_DIR}/")
    print(f"Saved charts (confusion_matrix.png, metrics.png, class_balance.png) to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
