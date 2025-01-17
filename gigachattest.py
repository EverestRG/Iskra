import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

torch.set_num_threads(12)

# Конфигурация
MODEL_NAME = "RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4"
MODEL_DIR = "./models/qwen"
DEFAULT_SYSTEM_PROMPT = "Ты — HALIS, русскоязычный автоматический ассистент для умного дома. Ты разговариваешь с людьми и помогаешь им, но делаешь это с сарказмом."

# Создаём папку для модели, если её нет
os.makedirs(MODEL_DIR, exist_ok=True)

# Проверяем, есть ли модель в локальной папке
if os.path.exists(os.path.join(MODEL_DIR, "config.json")):  # Проверяем наличие конфига
    print(f"Модель найдена в {MODEL_DIR}. Загружаем из локальной папки...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        #load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        #low_cpu_mem_usage=True,
        cache_dir="E:/.hfcache"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, cache_dir="E:/.hfcache")
else:
    print(f"Модель не найдена в {MODEL_DIR}. Скачиваем и сохраняем...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        #load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        #low_cpu_mem_usage=True,
        cache_dir="E:/.hfcache"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="E:/.hfcache")

    # Сохраняем модель и токенизатор в локальную папку
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Модель сохранена в {MODEL_DIR}.")

# Настройка генерации
model.eval()
generation_config = GenerationConfig.from_pretrained(MODEL_NAME, cache_dir="E:/.hfcache")
print(generation_config)

# Примеры запросов
inputs = ["Почему трава зеленая?", "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"]
for query in inputs:
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": query
    }], tokenize=False, add_generation_prompt=True)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True)
    data = {k: v.to(model.device) for k, v in data.items()}
    #data = {k: v.repeat(4, 1) for k, v in data.items()}

    # Создаём стример для потокового вывода
    streamer = TextStreamer(tokenizer, skip_prompt=True)  # skip_prompt=True, чтобы не выводить промпт

    # Генерация с потоковым выводом
    print(f"Вопрос: {query}")
    print("Ответ:")
    output_ids = model.generate(**data, generation_config=generation_config, max_new_tokens=500, streamer=streamer)
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print("\n==============================\n")