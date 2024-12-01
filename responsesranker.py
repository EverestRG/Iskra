from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
import os, re

def load_rank_model(model_path='./models/rut5'):
    """
    Загружает модель ранжирования из локальной папки, либо загружает её из интернета и сохраняет в указанную папку.
    model_path: путь к папке с моделью
    """
    # Проверяем, существует ли папка с моделью
    if not os.path.exists(model_path):
        print(f"Модель не найдена в '{model_path}', выполняется загрузка...")
        os.makedirs(model_path, exist_ok=True)

        # Указываем имя предобученной модели
        model_name = "cointegrated/rut5-base"

        # Загружаем модель и токенайзер
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Сохраняем модель в папку
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Модель успешно загружена и сохранена в '{model_path}'.")
    else:
        print(f"Загрузка модели из '{model_path}'...")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Модель успешно загружена из локальной директории.")

    return model, tokenizer


# Пример вызова функции
rank_model, rank_tokenizer = load_rank_model()

def rank_responses(prompt, responses, conversation_history=None):
    """
    Ранжирует ответы на основе релевантности к запросу.
    prompt: исходный запрос
    responses: список сгенерированных ответов
    """
    scores = []
    for response in responses:
        # Формируем ввод для модели ранжирования
        if conversation_history:
            input_text = f"контекст: {conversation_history} запрос: {prompt} ответ: {response}"
        else:
            input_text = f"Оцени релевантность: запрос: {prompt} ответ: {response}."
        #input_text = f"{prompt}\n{response}"
        inputs = rank_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

        # Генерация оценки качества
        output = rank_model.generate(
            **inputs,
            max_length=10,
            repetition_penalty=1.2,
            pad_token_id=rank_tokenizer.pad_token_id,
            eos_token_id=rank_tokenizer.eos_token_id,
            num_return_sequences=1
        )
        score = rank_tokenizer.decode(output[0], skip_special_tokens=True)
        print(score)

        # Конвертируем текстовый ответ в число
        scores.append(len(score))

    # Возвращаем ответ с максимальным "качеством"
    best_response_index = scores.index(max(scores))
    print(responses)
    return responses[best_response_index]