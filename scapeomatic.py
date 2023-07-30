import requests
from bs4 import BeautifulSoup
import os

# Specify the URL of the page
url = "https://www.osha.gov/publications/all"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}

# Send HTTP request to the specified URL and save the response from server in a response object called r
r = requests.get(url, headers=headers)

# Create a BeautifulSoup object and specify the parser library at the same time
soup = BeautifulSoup(r.text, 'html.parser')

# Find all the links on the webpage
links = soup.find_all('a')

# From the links, find the PDF files of interest
pdf_links = [link['href'] for link in links if link['href'].endswith('pdf') and 'English' in link.text]

# Download the PDFs
for link in pdf_links:
    response = requests.get(link, headers=headers)
    file_name = os.path.join('path/to/save/files', link.split('/')[-1])  # specify the path where you want to save the files
    with open(file_name, 'wb') as pdf_file:
        pdf_file.write(response.content)
