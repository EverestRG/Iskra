import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Путь к локальной модели
MODEL_PATH = "./models/emotionalanalysis"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Предобученная модель для анализа тональности

def load_model():
    """
    Загружает модель для анализа тональности.
    Если локальная модель отсутствует, загружает её и сохраняет в указанную папку.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Emotion recognition model not found at '{MODEL_PATH}', downloading...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        # Загружаем модель и сохраняем локально
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
    else:
        print(f"Loading emotion recognition model from '{MODEL_PATH}'")
    # Загружаем модель из локальной папки
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Инициализация пайплайна
sentiment_analyzer = load_model()

def calculate_happiness_index(text):
    """
    Использует локально установленную модель для анализа тональности текста
    и вычисляет индекс радости текста от -1 до 1.

    Параметры:
        text (str): Входной текст

    Возвращает:
        float: Индекс радости
    """
    # Анализируем тональность текста
    results = sentiment_analyzer(text)

    # Рассчитываем индекс на основе вероятностей
    for result in results:
        label = result['label']  # 'POSITIVE' или 'NEGATIVE'
        score = result['score']  # Вероятность
        print(label, score)
        if label == "POSITIVE":
            return score  # Чем выше вероятность, тем ближе к 1
        elif label == "NEGATIVE":
            return -score  # Чем выше вероятность, тем ближе к -1

    # Если текст не распознан, возвращаем 0 (нейтральное)
    return 0.0