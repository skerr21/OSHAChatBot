import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# Check if a GPU is available and if not, fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the models and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
question_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)
answer_pipeline = pipeline('question-answering', model='google/flan-t5-base', device=0 if device=='cuda' else -1)  # device=0 uses the first GPU

def generate_question(text):
    # Split the text into chunks of 512 tokens
    chunk_size = 512
    words = text.split(' ')
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    questions = []
    for chunk in chunks:
        input_text = "question: " + chunk
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        outputs = question_model.generate(input_ids)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions.append(question)
    
    return questions


def answer_question(questions, context):
    answers = []
    for question in questions:
        result = answer_pipeline({
            'question': question,
            'context': context
        })
        answers.append(result['answer'])
    return answers


# Specify the directory containing the text files
input_directory = 'pdfs'

# Specify the output file
output_file = 'questions_and_answers.txt'

# Loop over the text files
with open(output_file, 'w') as outfile:
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            # Extract the text from the file
            with open(os.path.join(input_directory, filename), 'r') as infile:
                text = infile.read()

            # Generate questions based on the text
            questions = generate_question(text)

            # Generate answers to the questions
            answers = answer_question(questions, text)

            # Write the questions and answers to the output file
            outfile.write(f"File: {filename}\n")
            for question, answer in zip(questions, answers):
                outfile.write(f"Question: {question}\n")
                outfile.write(f"Answer: {answer}\n")
            outfile.write("\n")
