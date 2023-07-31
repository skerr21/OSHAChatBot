from transformers import AutoModelForCausalLM, AutoTokenizer, LineByLineTextDataset, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AdamW, get_linear_schedule_with_warmup

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="all_preprocessed_text.txt",
    block_size=256,
)

# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=6,  # Increase the number of training epochs
    per_device_train_batch_size=4,  # Increase the batch size
    # gradient_accumulation_steps=16,  # Adjust gradient accumulation steps according to your new batch size
    fp16=False,  # Use full-precision training
    save_steps=10_000,
    save_total_limit=2,
)
# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

num_training_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)



trainer.train()

# Save the model
trainer.save_model()

# Save the tokenizer
tokenizer.save_pretrained("./fine_tuned_model")
