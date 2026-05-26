"""
FastAPI endpoint для классификации тональности отзывов.

Запуск локально:
    uvicorn src.api.main:app --reload --port 8000

Примеры запросов:
    GET  http://localhost:8000/
    GET  http://localhost:8000/health
    POST http://localhost:8000/predict   body: {"text": "Отличный товар!"}
"""

import pickle
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Sentiment Classifier API",
    description="Классификация тональности отзывов: negative / neutral / positive",
    version="1.0.0",
)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

model = None
vectorizer = None


@app.on_event("startup")
def load_model():
    global model, vectorizer
    model_path = os.getenv("MODEL_PATH", "checkpoints/sklearn_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
            model = saved["model"]
            vectorizer = saved["vectorizer"]
        print(f"Модель загружена из {model_path}")
    else:
        print(f"Модель не найдена по пути {model_path}")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"status": "ok", "service": "sentiment-classifier"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    text_clean = request.text.lower().strip()
    X = vectorizer.transform([text_clean])
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(proba[label])

    return PredictResponse(
        label=label,
        sentiment=LABEL_MAP[label],
        confidence=round(confidence, 4),
    )
