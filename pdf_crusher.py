from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader 
import os
from tqdm import tqdm
import threading

def pdf_to_text(pdf_file):
    print("Processing: ", pdf_file)
    pdf = PdfReader(pdf_file)
    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    return text.encode('utf-8', errors='ignore').decode()

pdf_folder = r'F:\ConversationalOsha\osha_pdfs'
output_file = 'output.txt'

full_text = [] 

with ThreadPoolExecutor(max_workers=8) as executor:

    futures = [executor.submit(pdf_to_text, os.path.join(pdf_folder, pdf))
               for pdf in os.listdir(pdf_folder) if pdf.endswith('.pdf')]

    with tqdm(total=len(futures)) as progress:
        for i, future in enumerate(as_completed(futures)):
            text = future.result()

            full_text.append(text)
            
            # Save progress every 100 files
            if i % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(full_text))
                    print('Saved progress')

            progress.update(1)

# Final save            
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(full_text))

print('Completed!')
