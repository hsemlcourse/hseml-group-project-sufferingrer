# ML Project — Классификация тональности отзывов на товары

**Студент:** Лелеко Дмитрий

**Группа:** БИВ238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** многоклассовая классификация тональности (3 класса: негативный / нейтральный / позитивный)

**Датасет:** [ai-forever/ru-reviews-classification](https://huggingface.co/datasets/ai-forever/ru-reviews-classification)

Датасет с HuggingFace, содержит 60 000 строк, 2 столбца (`text`, `label`), классы сбалансированы.

**Целевая переменная:** label (0 — негативный, 1 — нейтральный, 2 — позитивный)

**Метрика:** Macro F1


## Структура репозитория

```
.
├── data
│   ├── raw/                    # Исходные данные
│   ├── processed/              # Очищенные данные: train.csv, val.csv, test.csv
│   └── parsed/                 # Дополнительные данные (опционально)
├── notebooks
│   ├── 01_eda.ipynb            # Разведочный анализ
│   ├── 02_preprocessing.ipynb  # Очистка и подготовка данных
│   └── 03_baseline.ipynb       # Baseline-модель
├── src
│   ├── data/                   # Загрузка и очистка
│   ├── models/                 # Обучение моделей
│   └── api/                    # FastAPI
├── app/                        # Streamlit интерфейс
├── report
│   └── report.md
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные

- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные (train/val/test split)


## Результаты

### Результаты экспериментов

| Модель                              | Val Macro F1 | Примечание |
|-------------------------------------|--------------|----------|
|  Baseline: LogReg + TF-IDF        | 0.7487    | ngram(1,2), max_features=50k, sublinear_tf=True |
|  LogReg (GridSearch)              | 0.7488    | Лучшая модель. C=0.5, class_weight='balanced' |
|  LogReg + Word + Char n-grams     | 0.7438    | Комбинированные признаки (TF-IDF word + char_wb 3-6) |
|  Voting (LogReg + LightGBM + CatBoost)  | 0.7483 | Soft voting лучших моделей |
|  LinearSVC (calibrated)           | 0.7361       | С калибровкой вероятностей |
|  CatBoost + TF-IDF (GPU)          | 0.7294       | iterations=500-800, depth=6-8 |
|  LightGBM + TF-IDF (GPU)          | 0.7227       | n_estimators=500-800, num_leaves=63-127 |
|  Random Forest                    | 0.7165       | n_estimators=200, class_weight='balanced' |
|  KNN (cosine)                     | 0.6621       | n_neighbors=7 |

Финальной моделью выбрана Logistic Regression после GridSearch (C=0.5, class_weight='balanced'), так как она показала наивысший результат по Macro F1 (0.7488)

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
