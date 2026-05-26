"""
Streamlit интерфейс для классификации тональности отзывов.

Запуск:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

EMOJI = {"negative": "😡", "neutral": "😐", "positive": "😊"}
COLOR = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}
LABEL_RU = {"negative": "Негативный", "neutral": "Нейтральный", "positive": "Позитивный"}
TEMPLATES = {
    "negative": (
        "Здравствуйте! Приносим извинения за доставленные неудобства."
        "Пожалуйста, напишите нам в личные сообщения — разберёмся и решим вопрос."
    ),
    "neutral": (
        "Здравствуйте! Спасибо за ваш отзыв. "
        "Если у вас есть вопросы или пожелания — всегда рады помочь."
    ),
    "positive": (
        "Здравствуйте! Большое спасибо за тёплый отзыв!"
        "Рады, что товар вам понравился. Ждём вас снова!"
    ),
}

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

st.title("Классификатор тональности отзывов")
st.markdown("Вставьте отзыв на товар — получите тональность и готовый шаблон ответа.")

text = st.text_area(
    "Текст отзыва",
    height=150,
    placeholder="Например: Товар пришёл быстро, качество отличное!",
)

if st.button("Определить тональность", type="primary"):
    if not text.strip():
        st.warning("Введите текст отзыва")
    else:
        try:
            resp = requests.post(API_URL, json={"text": text}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            sentiment = data["sentiment"]
            confidence = data["confidence"]

            st.markdown(
                f"""
                <div style='background:{COLOR[sentiment]}22; border-left:4px solid {COLOR[sentiment]};
                            padding:16px; border-radius:8px; margin:12px 0'>
                    <h3 style='color:{COLOR[sentiment]}; margin:0'>
                        {EMOJI[sentiment]} {LABEL_RU[sentiment]}
                    </h3>
                    <p style='margin:4px 0 0'>Уверенность модели: <b>{confidence * 100:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Шаблон ответа")
            st.text_area(
                "Готовый ответ:",
                value=TEMPLATES[sentiment],
                height=100,
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "API недоступен. Запустите сервер: "
                "`uvicorn src.api.main:app --port 8000`"
            )
        except Exception as e:
            st.error(f"Ошибка: {e}")

st.divider()
st.caption("Модель: LogReg + TF-IDF | Датасет: ai-forever/ru-reviews-classification")
