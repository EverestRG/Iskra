import os
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch, re

print("Loading model...")

app = Flask(__name__)

# Путь к моделям
gpt2_model_path = "./models/gpt2"
dialo_model_path = "./models/dialo"
dialol_model_path = "./models/dialolarge"
savepth = "./save.txt"
dialog_history = []

try:
    with open(savepth, 'r') as save:
        #dialog_history = save.read().split('#$%')
        print('Readed dialog history')
except:
    with open(savepth, 'w+') as save:
        save.close()

try:
    dialog_history.pop(dialog_history.index(''))
except: None

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


# Функция для генерации текста
def generate_text(prompt):
    # Добавляем последний запрос в историю
    # Ограничиваем историю последними 5 сообщениями

    # Ограничиваем количество токенов в контексте, чтобы избежать переполнения
    max_length = 1024  # Максимальное количество токенов для GPT-2/ DialoGPT
    inputs = tokenizer.encode(
        '<history>' + "Hi! I'm your father and you are my kind and clever daughter. Your name is Iskra. " +
        "You always remember your name and respond to prompts based on your knowledge and the conversation history. " +
        tokenizer.eos_token.join(dialog_history) +
        tokenizer.eos_token +
        '<prompt>' + prompt +
        tokenizer.eos_token,
        return_tensors="pt", truncation=True, max_length=max_length
    )
    print(f"prompt: {tokenizer.decode(inputs[0], skip_special_tokens=False)}")
    #inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length)

    dialog_history.append(prompt)
    if len(dialog_history) > 2:
        dialog_history.pop(0)

    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    outputs = model.generate(
        inputs,
        max_length=300,  # Общая длина (включая входной контекст)
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.5,  # Более стабильные ответы
        top_k=75,  # Умеренное разнообразие
        top_p=0.75,  # Сбалансированный уровень вероятности
        max_new_tokens=25,  # Ограничиваем длину ответа
        repetition_penalty=1.3,  # Уменьшает бред
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True  # Включаем сэмплирование для разнообразия
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Обрезаем повторение запроса в ответе (убираем вопрос из начала ответа)
    text = text.replace(tokenizer.decode(inputs[0], skip_special_tokens=True), '')

    text = text.replace('\n', ' ').replace('...', '.').strip()
    text = re.sub(r'[^\w\s.,!?]', '', text)

    if '<|endoftext|>' in text:
        text = text.split('<|endoftext|>')[0]

    # Добавляем сгенерированный ответ в историю для дальнейшего контекста
    dialog_history[-1] = dialog_history[-1] + ' -> ' + text
    print(f"Generated text: {text}")
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
    with open(savepth, 'w+') as save:
        save.write('#$%'.join(dialog_history))
