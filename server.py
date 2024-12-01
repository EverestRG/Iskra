import os
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
#from emotion_index import calculate_happiness_index
import torch, re, random
from contextshift import detect_context_shift
from  responsesranker import rank_responses

print("Loading model...")

app = Flask(__name__)

# Путь к моделям
gpt2_model_path = "./models/gpt2"
dialo_model_path = "./models/dialo"
dialol_model_path = "./models/dialolarge"
rugpt_model_path = "./models/rugpt"

# Выбор модели: gpt2 или dialo
model_choice = "rugpt"  # Можно менять на "gpt2" или "dialo"

# Функции для сохранения и загрузки истории
def load_user_history(user_id):
    # Загрузка истории сообщений
    history_file = os.path.join('history', f'{user_id}.txt')
    token_history_file = os.path.join('token_history', f'{user_id}.txt')

    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            user_history = f.read().split('%NEXT%')
    else:
        user_history = []

    user_token_history = []

    if os.path.exists(token_history_file):
        with open(token_history_file, 'r', encoding='utf-8') as f:
            readed = f.readlines()
            for i in range(len(readed)):
                user_token_history.append(tokenizer.encode(readed[i], return_tensors="pt", truncation=True, max_length=1024))

    return user_history, user_token_history

def save_user_history(user_id, user_history, user_token_history):
    # Сохранение истории сообщений
    os.makedirs('history', exist_ok=True)
    os.makedirs('token_history', exist_ok=True)

    history_file = os.path.join('history', f'{user_id}.txt')
    token_history_file = os.path.join('token_history', f'{user_id}.txt')

    with open(history_file, 'w', encoding='utf-8') as f:
        f.write('%NEXT%'.join(user_history))

    with open(token_history_file, 'w', encoding='utf-8') as f:
        story = []
        for i in range(len(user_token_history)):
            story.append(tokenizer.decode(user_token_history[i][0], skip_special_tokens=True).replace('\n', ''))
        while True:
            try:
                story.pop(story.index(''))
            except:
                break
        f.write('\n'.join(story))

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
    elif model_choice == "rugpt":
        model_path = rugpt_model_path
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
        elif model_choice == "rugpt":
            model_name = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
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
    short_word_exceptions = {"-", "у", "ок", "хе", "хи", "да", "а", "ха", "я", "ты", "те", "ту", "то", "он", "не", "но", "и", "по", "с", "со", "на", "от", "за", "в"}

    filtered_words = [
        word for word in text.split()
        if len(word.replace('.', '').replace('!', '').replace('?', '')) > 2 or word.lower() in short_word_exceptions
    ]

    # Собираем текст обратно
    #text = ' '.join(filtered_words)

    text = text.replace(')))', '')
    text = text.replace(' ) ', '')
    text = text.replace('(((', '')
    text = text.replace(' ( ', '')

    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Функция для добавления эмоции в ответ
def add_emotion(text, emotion=0.0):
    print(emotion)
    if emotion > 0.99979:
        return text + ' ' + random.choice(["😃", "😄", "😊"])
    if emotion < -0.99979:
        return text + ' ' + random.choice(["🙁", "😥", "😖"])
    return text


# Функция для обрезки текста до последнего завершённого предложения
def trim_to_sentence(text):
    last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))  # Ищем последний знак препинания
    if last_punctuation != -1:
        return text[:last_punctuation + 1].strip()  # Возвращаем текст до последнего знака
    return text.strip()  # Если знаков препинания нет, возвращаем весь текст


# Функция для генерации текста
def generate_text(prompt, user_history, user_token_history):
    global model, tokenizer

    # Проверка на команду сброса истории
    if prompt == "%%<|RESET|>%%":
        user_history.clear()  # Очищаем историю сообщений
        user_token_history.clear()  # Очищаем историю токенов
        return "%|DONE|%", user_history, user_token_history  # Отправляем подтверждение сброса

    # Формируем статический контекст
    static_instructions = "User1: Привет, я Искра! Чем могу помочь?"

    def txtlen(string: str) -> str:
        length = len(tokenizer.encode(string))
        if length <= 15:
            return '1'
        if length <= 50:
            return '2'
        if length <= 256:
            return '3'
        return '-'

    shifted = False

    if len(user_token_history) > 0:
        outs = []
        for i in range(len(user_token_history)):
            tt = tokenizer.decode(user_token_history[i][0], skip_special_tokens=True).replace('\n', '')
            if tt.endswith(('.', '!', '?')):
                outs.append(tt)
            else:
                outs.append(tt + '.')
        outs.append(prompt)
        shifted = detect_context_shift(outs, 0.42)

    if shifted:
        user_token_history.clear()

    # Формируем динамическую часть истории
    conversation_history = ""
    for i in range(0, len(user_token_history), 2):
        user_input = tokenizer.decode(user_token_history[i][0], skip_special_tokens=True).replace('\n', '')
        bot_response = tokenizer.decode(user_token_history[i + 1][0], skip_special_tokens=True).replace('\n', '') if i + 1 < len(user_token_history) else ""
        conversation_history += f"|0|{txtlen(user_input)}|{user_input}\n|1|{txtlen(bot_response)}||0|1|{bot_response}\n"

    clean_context = conversation_history
    conversation_history += f"|0|{txtlen(prompt)}|{prompt}{tokenizer.eos_token}|1|2|"

    context = conversation_history

    #context = f"{prompt + tokenizer.eos_token}"

    # Токенизация контекста
    inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=1024)

    # Добавляем текущий запрос в историю
    user_token_history.append(tokenizer.encode(f"{prompt}", return_tensors="pt"))

    # Ограничиваем длину истории
    if len(user_token_history) > 4:  # 1 пара (вопрос-ответ)
        user_token_history.pop(0)
        user_token_history.pop(0)

    attention_mask = torch.ones(inputs.shape, device=inputs.device)
    #attention_mask = (inputs != tokenizer.pad_token_id).long()

    print(context)

    # Генерация текста
    outputs = model.generate(
        inputs,
        max_length=256,
        early_stopping=True,
        num_return_sequences=2,
        no_repeat_ngram_size=3,
        num_beams=4,
        temperature=0.65,
        top_k=70,
        top_p=0.85,
        max_new_tokens=100,
        repetition_penalty=1.4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True
    )

    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    responsesclean = [response.replace(context.replace(tokenizer.eos_token, '').replace(tokenizer.bos_token, ''), '') for response in responses]

    # Декодируем текст
    #if not clean_context == "":
    #    text = rank_responses(prompt, responsesclean, clean_context)
    #else:
    #    text = rank_responses(prompt, responsesclean)

    text = rank_responses(prompt, responsesclean)

    # Чистим текст
    text = clean_generated_text(text)
    #text = trim_to_sentence(text)

    # Добавляем ответ в историю
    response_tokens = tokenizer.encode(f"{text}", return_tensors="pt")
    user_token_history.append(response_tokens)

    # Добавляем эмоцию
    #if random.choice([0, 1, 1, 1]):
    #    text = add_emotion(text, calculate_happiness_index(text))

    # Добавляем в историю диалога
    user_history.append(f"[You]: {prompt}\n\n[Iskra]: {text}")

    text = '\n\n'.join(user_history)

    return text, user_history, user_token_history

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    prompt = data.get('prompt', '')
    user_id = data.get('user_id')  # Получаем уникальный ID пользователя

    if not user_id:
        return jsonify({"response": "User ID is required"}), 400

    # Загружаем историю пользователя
    user_history, user_token_history = load_user_history(user_id)

    # Генерация ответа с использованием истории этого пользователя
    response_text, user_history, user_token_history = generate_text(prompt, user_history, user_token_history)

    # Сохраняем обновленную историю пользователя
    save_user_history(user_id, user_history, user_token_history)

    return jsonify({"response": response_text}), 400

@app.route('/get_history', methods=['POST'])
def get_history():
    data = request.get_json()
    user_id = data.get('user_id')  # Получаем уникальный ID пользователя

    if not user_id:
        return jsonify({"response": "User ID is required"}), 400

    # Загружаем историю пользователя
    user_history, user_token_history = load_user_history(user_id)

    return jsonify({"response": "\n\n".join(user_history)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
