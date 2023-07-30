import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

directory = 'pdfs'
max_length = 100  # Define your desired maximum sequence length

# Initialize a list to store all preprocessed sentences
all_sentences = []

# Process each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            text = file.read()

        # Lowercase the text
        text = text.lower()

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = re.sub(r'\W', ' ', text)

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Process each sentence
        # Process each sentence
        for sentence in sentences:
            # Remove excessive white spaces
            sentence = re.sub(r'\s+', ' ', sentence)

            # Tokenize the sentence
            tokens = word_tokenize(sentence)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [i for i in tokens if not i in stop_words]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # Split long sentences into multiple shorter sequences
            for i in range(0, len(tokens), 250):  # Set maximum sequence length to 400 words
                all_sentences.append(' '.join(tokens[i:i+400]))


# Save the preprocessed sentences to a new file, each on a separate line
with open('all_preprocessed_text.txt', 'w') as file:
    file.write('\n'.join(all_sentences))
