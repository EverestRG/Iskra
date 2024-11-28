import os
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch, re, random

print("Loading model...")

app = Flask(__name__)

# Путь к моделям
gpt2_model_path = "./models/gpt2"
dialo_model_path = "./models/dialo"
dialol_model_path = "./models/dialolarge"
savepth = "./save.txt"
dialog_history = []
dialog_history_tokens = []

# Выбор модели: gpt2 или dialo
model_choice = "dialolarge"  # Можно менять на "gpt2" или "dialo"

# Проверяем, есть ли папка с моделью
def load_model(model_choice):
    if model_choice == "gpt2":
        model_path = gpt2_model_path
        model_class = GPT2LMHeadModel
        tokenizer_class = GPT2Tokenizer
    elif model_choice == "dialo":
        model_path = dialo_model_path
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
    elif model_choice == "dialolarge":
        model_path = dialol_model_path
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
    else:
        raise ValueError("Unsupported model choice")

    # Проверяем, существует ли папка с моделью
    if not os.path.exists(model_path):
        print(f"Model folder '{model_path}' not found, downloading...")
        if model_choice == "dialo":
            model_name = "microsoft/DialoGPT-medium"  # Можно использовать 'microsoft/DialoGPT-large' или 'microsoft/DialoGPT-small'
        elif model_choice == "gpt2":
            model_name = "gpt2-medium"
        elif model_choice == "dialolarge":
            model_name = "microsoft/DialoGPT-large"
        model = model_class.from_pretrained(model_name)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        # Сохраняем модель в папку
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        print(f"Loading model from {model_path}...")
        model = model_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(model_path)

    return model, tokenizer


# Загружаем модель
model, tokenizer = load_model(model_choice)

print("Model loaded!")


# Функция для очистки текста от лишних символов
def clean_generated_text(text):
    """
    Убирает случайные символы, повторяющиеся буквы и удаляет короткие слова,
    кроме часто используемых исключений.
    """
    # Исключения для коротких слов
    short_word_exceptions = {"i", "im", "ur", "am", "on", "of", "an", "a", "it", "is", "in", "at", "to", "by", "as", "he", "we", "do", "be", "go", "me", "my", "no", "or", "up", "us"}

    # Убираем случайные символы
    text = re.sub(r'[^\w\s.,!?]', '', text)

    # Удаляем короткие слова, кроме исключений
    filtered_words = [
        word for word in text.split()
        if len(word) > 2 or word.lower() in short_word_exceptions
    ]

    # Собираем текст обратно
    text = ' '.join(filtered_words)

    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Функция для определения эмоций на основе текста
previous_emotion = "neutral"


def detect_emotion(prompt, previous_emotion="neutral"):
    """
    Определяет эмоцию, учитывая как текущий запрос, так и предыдущее состояние.
    Если эмоция явно изменяется, она обновляется.
    """
    # Словарь ключевых слов для эмоций
    sad_keywords = ["sad", "down", "unhappy", "blue", "gloomy", "depressed"]
    happy_keywords = ["happy", "joy", "great", "good", "excited", "fun", "smile", "bright"]
    neutral_keywords = ["okay", "fine", "normal", "alright"]

    # Если в запросе есть ключевые слова, указывающие на грусть или радость, обновляем состояние
    if any(word in prompt.lower() for word in sad_keywords):
        return "sad"
    elif any(word in prompt.lower() for word in happy_keywords):
        # Если запрос содержит слова о счастье, но текущая эмоция грустная, не меняем её
        if previous_emotion == "sad":
            return previous_emotion
        return "happy"
    elif any(word in prompt.lower() for word in neutral_keywords):
        return "neutral"

    # Если ключевые слова противоречат предыдущему состоянию, сохраняем текущее состояние
    return previous_emotion


# Функция для добавления эмоции в ответ
def add_emotion(text, emotion="happy"):
    emotions = {
        "happy": ["😊", "😄"],
        "excited": ["😄"],
        "curious": ["🤔"],
        "thoughtful": ["🤔"],
        "neutral": [""],
        "sad": ["😢"]
    }
    if emotion in emotions:
        return text + " " + random.choice(emotions[emotion])
    return text


# Функция для обрезки текста до последнего завершённого предложения
def trim_to_sentence(text):
    last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))  # Ищем последний знак препинания
    if last_punctuation != -1:
        return text[:last_punctuation + 1].strip()  # Возвращаем текст до последнего знака
    return text.strip()  # Если знаков препинания нет, возвращаем весь текст


# Функция для генерации текста
def generate_text(prompt):
    global previous_emotion, dialog_history_tokens, dialog_history

    # Определяем текущую эмоцию
    current_emotion = detect_emotion(prompt, previous_emotion)

    # Формируем статический контекст (монолог от лица System)
    static_instructions = (
        f"<|user|> Your task is to respond to my queries clearly with kindness.\n<|iskra|> Understood.{tokenizer.eos_token}"
        #f"\nCurrently, you feel {current_emotion}."
    )

    # Формируем динамическую часть истории (история диалога)
    conversation_history = ""
    for i in range(0, len(dialog_history_tokens), 2):  # Чередуем запросы и ответы
        user_input = tokenizer.decode(dialog_history_tokens[i][0], skip_special_tokens=True)
        bot_response = tokenizer.decode(dialog_history_tokens[i + 1][0], skip_special_tokens=True) if i + 1 < len(dialog_history_tokens) else ""
        conversation_history += f"\n<|user|> {user_input}\n<|iskra|> {bot_response}{tokenizer.eos_token}"

    # Добавляем текущий запрос
    conversation_history += f"\n<|user|> {prompt}"

    # Объединяем контекст
    #context = "".join([static_instructions, conversation_history])

    context = conversation_history

    print(context)

    # Токенизация контекста
    inputs = tokenizer.encode(context + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=1024)
    #inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=1024)

    # Добавляем текущий запрос в историю
    dialog_history_tokens.append(tokenizer.encode(f"{prompt}", return_tensors="pt"))

    # Ограничиваем длину истории
    if len(dialog_history_tokens) > 2:  # 3 пары (вопрос-ответ)
        dialog_history_tokens.pop(0)
        dialog_history_tokens.pop(0)

    # Маска внимания
    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    # Генерация текста
    outputs = model.generate(
        inputs,
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.11,
        top_k=29,
        top_p=0.7,
        max_new_tokens=25,
        repetition_penalty=1.6,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True
    )

    # Декодируем текст
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Убираем повторение запроса из ответа
    text = text.replace(tokenizer.decode(inputs[0], skip_special_tokens=True), '')

    # Чистим текст
    text = clean_generated_text(text)
    text = trim_to_sentence(text)

    # Добавляем ответ в историю
    response_tokens = tokenizer.encode(f"{text}", return_tensors="pt")
    dialog_history_tokens.append(response_tokens)

    # Добавляем эмоцию
    text = add_emotion(text, current_emotion)

    # Сохраняем текущую эмоцию
    previous_emotion = current_emotion

    print(f"Generated text: {text}")

    dialog_history.append(f"{prompt} -> {text}")

    return text

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    prompt = data.get('prompt', '')
    # Генерация ответа
    if prompt:
        response_text = generate_text(prompt)
        return jsonify({"response": response_text})
    else:
        return jsonify({"response": "No prompt provided"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    print('Saving dialog context')
    with open(savepth, 'w+', encoding="utf-8") as save:
        save.write('\n'.join(dialog_history))