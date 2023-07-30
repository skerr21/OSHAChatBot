from transformers import AutoModelForCausalLM, AutoTokenizer, LineByLineTextDataset, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
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
    num_train_epochs=1,  # Increase the number of training epochs
    per_device_train_batch_size=1,  # Increase the batch size
    gradient_accumulation_steps=1,  # Adjust gradient accumulation steps according to your new batch size
    fp16=True,  # Use full-precision training
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

trainer.train()

# Save the model
trainer.save_model()

# Save the tokenizer
tokenizer.save_pretrained("./fine_tuned_model")
