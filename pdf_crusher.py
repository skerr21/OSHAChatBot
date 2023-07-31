import os
import re
from pathlib import Path

from pdfminer.high_level import extract_text
import nltk
from nltk.tokenize import word_tokenize

input_folder = 'osha_pdfs'
output_file = 'cleaned_text.txt'

combined_text = ""

for pdf_path in Path(input_folder).glob('*.pdf'):
    text = extract_text(pdf_path)
    
    # Clean text
    text = re.sub(r'\n\s*\n', '\n', text) # Remove blank lines
    text = re.sub(r'[\n\t\s]+', ' ', text) # Replace newlines, tabs, multiple spaces with single space
    
    combined_text += text + "\n"
    
with open(output_file, 'w') as f:
    f.write(combined_text)
    
# Tokenize
tokens = word_tokenize(combined_text)

print(f"Saved {len(tokens)} tokens to {output_file}")
