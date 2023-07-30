from transformers import AutoModelForCausalLM, AutoTokenizer, LineByLineTextDataset, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import nltk
import re
nltk.download('punkt')
# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer.pad_token = tokenizer.eos_token
# Make sure gradient_checkpointing is enabled to save memory
model.config.gradient_checkpointing = True

class ConversationPairsDataset(Dataset):
    def __init__(self, tokenizer, text, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Split the text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Pair each sentence with the following sentence
        self.examples = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        # Tokenize the pair of sentences and return as a single example
        return self.tokenizer(self.examples[i][0] + ' ' + self.examples[i][1], truncation=True, max_length=self.block_size, padding='max_length')
# Prepare the dataset
with open("All About OSHA - all_about_OSHA.txt", 'r', encoding='utf-8') as f:
    text = f.read()

dataset = ConversationPairsDataset(
    tokenizer=tokenizer,
    text=text,
    block_size=128  # Reduce the block size if necessary
)


# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduced batch size
    gradient_accumulation_steps=16,  # Add gradient accumulation
    fp16=True,  # Enable half-precision training
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

