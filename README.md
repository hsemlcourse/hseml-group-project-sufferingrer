# Ozon Sentiment — Классификация тональности отзывов

Автоматическое определение тональности (positive / negative / neutral) пользовательских отзывов на товары. Основной датасет — [ai-forever/ru-reviews-classification](https://huggingface.co/datasets/ai-forever/ru-reviews-classification). Финальная модель — fine-tuned [rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2).

## Цель
Инструмент для селлеров на Ozon: автоматически классифицировать входящий отзыв и подбирать шаблон ответа — без ручной работы и без переплаты за встроенный AI Ozon (1.5% от оборота).

## Структура проекта

```
ozon-sentiment/
├── data/
│   ├── raw/                  # сырые данные (не коммитим в git)
│   ├── processed/            # после очистки: train.csv, val.csv, test.csv
│   └── parsed/               # спарсенные отзывы с Ozon (опционально)
│
├── notebooks/
│   ├── 01_eda.ipynb          # разведочный анализ
│   ├── 02_preprocessing.ipynb# очистка, фичи, сплит
│   ├── 03_baseline.ipynb     # LogReg + TF-IDF baseline
│   └── 04_experiments.ipynb  # эксперименты с моделями (CP2)
│
├── src/
│   ├── data/
│   │   ├── loader.py         # загрузка датасета
│   │   ├── cleaner.py        # функции очистки
│   │   └── parser.py         # парсинг Ozon (опционально)
│   ├── models/
│   │   ├── train_bert.py     # обучение rubert-tiny2 на GPU-сервере
│   │   └── baseline.py       # sklearn baseline
│   └── api/
│       └── main.py           # FastAPI endpoint (CP3)
│
├── app/
│   └── streamlit_app.py      # веб-интерфейс для селлера (CP3)
│
├── report/
│   └── report.md             # финальный отчёт
│
├── checkpoints/              # веса моделей (не коммитим, .gitignore)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .ruff.toml
└── README.md
```

## Быстрый старт

```bash
git clone <repo>
cd ozon-sentiment
pip install -r requirements.txt


### Docker (API + интерфейс)
```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

## Метрика
**Macro F1** — датасет сбалансирован (по 33% на каждый класс), 
поэтому Accuracy тоже валидна. Но выбираем Macro F1, потому что:
1. В реальных отзывах Ozon классы несбалансированы (негативных меньше)
2. Macro F1 одинаково штрафует за ошибки на каждом классе независимо от его размера
3. Это стандарт для задач классификации тональности в продакшне

## Результаты

| Модель | Val Macro F1 | Примечание |
|---|---|---|
| Baseline: LogReg + TF-IDF | 0.7481 | ngram (1,2), 50k features, C=1.0 |
| CatBoost + TF-IDF | — | CP2 |
| rubert-tiny2 fine-tune | — | CP2, 4 эпохи, lr=2e-5 |

## Чекпоинты

- **CP1**: EDA + preprocessing + baseline (`branch: cp1`)
- **CP2**: Эксперименты с моделями (`branch: cp2`)
- **CP3**: Деплой API + Streamlit (`branch: cp3`)
