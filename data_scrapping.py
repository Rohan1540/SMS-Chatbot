import re
from bs4 import BeautifulSoup
import requests
import time

''' Function to Clean the HTML page'''
def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'sup', 'table', 'nav', 'ol', 'ul']):
        element.decompose()
    
    # Remove references like [1], [2], etc.
    content = re.sub(r'\[\d+\]', '', soup.get_text())
    
    return ' '.join(content.split())

''' Function for data scrapping from Wikipedia '''
def scrape_wikipedia_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return clean_html(response.text)
    else:
        print(f"Failed to retrieve Wikipedia page: {url}")
        return ""

''' Function to scrape any additional links in the web page '''
def scrape_additional_pages(base_url, main_page_content):
    soup = BeautifulSoup(main_page_content, 'html.parser')
    links = soup.select('div.mw-parser-output a[href^="/wiki/"]')
    scraped_articles = []
    
    for link in links:
        href = link.get('href')
        full_url = f"https://en.wikipedia.org{href}"
        page_text = scrape_wikipedia_page(full_url)
        if page_text:
            scraped_articles.append(page_text + " <endoftext>\n")
    
    return scraped_articles


wikipedia_urls = {
    "Medicine": 'https://en.wikipedia.org/wiki/Medicine',
    "Diabetes": 'https://en.wikipedia.org/wiki/Diabetes',
    "Hypertension": 'https://en.wikipedia.org/wiki/Hypertension',
    "Cancer": 'https://en.wikipedia.org/wiki/Cancer',
    "Cardiology": 'https://en.wikipedia.org/wiki/Cardiology',
    "Neurology": 'https://en.wikipedia.org/wiki/Neurology',
    "Psychiatry": 'https://en.wikipedia.org/wiki/Psychiatry',
    "Pediatrics": 'https://en.wikipedia.org/wiki/Pediatrics',
    "Surgery": 'https://en.wikipedia.org/wiki/Surgery',
    "Radiology": 'https://en.wikipedia.org/wiki/Radiology',
    "Upper Respiratory Infection": 'https://en.wikipedia.org/wiki/Upper_respiratory_tract_infection',
    "Lower Respiratory Infection": 'https://en.wikipedia.org/wiki/Lower_respiratory_tract_infection',
    "Infectious Disease": 'https://en.wikipedia.org/wiki/Infectious_disease',
    "Mental Health": 'https://en.wikipedia.org/wiki/Mental_health',
    "Gastrointestinal Disease": 'https://en.wikipedia.org/wiki/Gastrointestinal_disease',
    "HIV/AIDS": 'https://en.wikipedia.org/wiki/HIV/AIDS',
    "Influenza": 'https://en.wikipedia.org/wiki/Influenza'
}

# Collect all articles
all_articles = []

# Scrape Wikipedia pages
for topic, url in wikipedia_urls.items():
    main_page_text = scrape_wikipedia_page(url)
    if main_page_text:
        all_articles.append(main_page_text + " <endoftext>\n")
        
        # Scrape additional related pages
        additional_articles = scrape_additional_pages(url, main_page_text)
        all_articles.extend(additional_articles)

# Save the cleaned text to a file
with open('scraped_articles_extended.txt', 'w', encoding='utf-8') as f:
    for article in all_articles:
        f.write(article + "\n")

print("Scraping and preprocessing complete. Data saved to scraped_articles_extended.txt")


''' Dedicated Functions to scrape a web page that consists of multiple links for the Neurological Disease '''
def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'sup', 'table', 'nav', 'ol', 'ul']):
        element.decompose()
    
    # Remove references like [1], [2], etc.
    content = re.sub(r'\[\d+\]', '', soup.get_text())
    
    return ' '.join(content.split())

def scrape_wikipedia_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return clean_html(response.text)
    else:
        print(f"Failed to retrieve Wikipedia page: {url}")
        return ""

def get_links_from_main_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.select('div.mw-parser-output a[href^="/wiki/"]')
        return [f"https://en.wikipedia.org{link.get('href')}" for link in links if ':' not in link.get('href')]
    else:
        print(f"Failed to retrieve the main page: {url}")
        return []

# URL of the main page
main_page_url = 'https://en.wikipedia.org/wiki/List_of_neurological_conditions_and_disorders'

# Get all the links from the main page
all_links = get_links_from_main_page(main_page_url)

# Collect all articles
all_articles = []

# Scrape each linked page
for url in all_links:
    page_text = scrape_wikipedia_page(url)
    if page_text:
        all_articles.append(page_text + " <endoftext>\n")
        time.sleep(0.5)  # To avoid hitting Wikipedia's rate limit

# Save the cleaned text to a file
with open('neurological_conditions_articles.txt', 'w', encoding='utf-8') as f:
    for article in all_articles:
        f.write(article + "\n")

print("Scraping and preprocessing complete. Data saved to neurological_conditions_articles.txt")