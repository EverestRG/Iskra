from sentence_transformers import SentenceTransformer, util
import os, torch

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIR = "./models/paraphrase-multilingual"


def load_model():
    if not os.path.exists(MODEL_DIR):
        print(f"Модель не найдена. Скачиваем {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_DIR)
    else:
        print(f"Модель найдена в {MODEL_DIR}. Загружаем...")
        model = SentenceTransformer(MODEL_DIR)
    return model

model = load_model()

def detect_context_shift(sentences, threshold=0.7):
    """
    Анализирует, насколько контекст нового предложения соответствует предыдущим.
    sentences: список строк (включая новое предложение)
    threshold: порог для определения смены контекста (0.7 = 70% схожести)
    """
    if len(sentences) < 2:
        return False  # Недостаточно данных для сравнения

    # Генерация эмбеддингов для предложений
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Сравнение последнего предложения со всеми предыдущими
    similarities = util.cos_sim(embeddings[-1], embeddings[:-1])

    # Проверяем, есть ли существенное изменение контекста
    max_similarity = torch.max(similarities).item()
    print(max_similarity)
    return max_similarity < threshold