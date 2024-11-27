try:
    with open("./dialogs.txt") as dial:
        rd = dial.read()
        if rd.replace('\n', '').replace(' ', '') == '':
            print('You must write dialogs for training in "dialogs.txt"!')
            exit(0)
except:
    with open("./dialogs.txt", "w+") as dial:
        None
    print('You must write dialogs for training in "dialogs.txt"!')
    exit(0)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

print("Loading model...")

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

print("Model loaded!")

# Подготовка данных
train_path = "./dialogs.txt"
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Настройка обучения
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-daughter",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print('Starting training...')

trainer.train()
model.save_pretrained("./gpt2-finetuned-daughter")
print("Trained successfully!")