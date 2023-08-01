from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AdamW, get_linear_schedule_with_warmup
from datasets import Dataset

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize the text
with open('cleaned_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text and convert tokens to IDs
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Split the token IDs into chunks of fixed size
max_input_length = model.config.max_position_embeddings  # max input length for the GPT-2 model
tokenized_texts = [token_ids[i:i+max_input_length] for i in range(0, len(token_ids), max_input_length)]

# Prepare the dataset
data = {'input_ids': tokenized_texts}
dataset = Dataset.from_dict(data)

# Check that the dataset is not empty
if len(dataset) == 0:
    raise ValueError("Dataset is empty.")

# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=9,  # Increase the number of training epochs
    per_device_train_batch_size=2,  # Increase the batch size
    # gradient_accumulation_steps=8,  # Adjust gradient accumulation steps according to your new batch size
    fp16=True,  # Use full-precision training
    save_steps=10_000,
    save_total_limit=2,
)

# Set up the optimizer and learning rate scheduler
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

trainer_temp = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

num_training_steps = len(trainer_temp.get_train_dataloader()) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

trainer_temp.train()
