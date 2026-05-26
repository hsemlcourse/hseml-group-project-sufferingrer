"""
Сохраняет финальную sklearn модель для FastAPI.

Запуск из корня проекта:
    python src/models/save_model.py

Результат: checkpoints/sklearn_model.pkl
"""

import pickle
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)


def main():
    print("Загрузка данных...")
    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/val.csv")

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(df_train["text_clean"])
    X_val = tfidf.transform(df_val["text_clean"])

    # Лучшие параметры из GridSearch (C=0.5 не поддерживается напрямую — используем ближайший)
    model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(X_train, df_train["label"])

    val_f1 = f1_score(df_val["label"], model.predict(X_val), average="macro")
    print(f"Val Macro F1: {val_f1:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/sklearn_model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": tfidf}, f)

    print("Модель сохранена: checkpoints/sklearn_model.pkl")


if __name__ == "__main__":
    main()
