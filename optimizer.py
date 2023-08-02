from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AdamW, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import torch


model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize text
with open('cleaned_text_connectors.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Split into chunks
max_input_length = model.config.max_position_embeddings  
tokenized_texts = [token_ids[i:i+max_input_length] for i in range(0, len(token_ids), max_input_length)]

# Create dataset
data = {'input_ids': tokenized_texts}
dataset = Dataset.from_dict(data)

if len(dataset) == 0:
    raise ValueError("Dataset is empty.")

# Data collator and training args
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir='./results_connectors',
    overwrite_output_dir=True, 
    num_train_epochs=6,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,  # Gradient accumulation
    fp16=True,  # Mixed precision training
    save_steps=10_000,
    save_total_limit=2,
)

# Optimizer and scheduler
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

# Apply PEFT to the model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    optimizers=(optimizer, lr_scheduler)  
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./tokenizer')
