import os
import re
import xml.etree.ElementTree as ET



    
def clean_text(text):
    if text is None:
        return ""
    
    # Remove XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special Wikipedia markup
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    
    # Remove references
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove double square brackets, keeping the text inside
    text = re.sub(r'\[\[([^\]|]+)\|?([^\]]*)\]\]', lambda m: m.group(2) or m.group(1), text)
    
    # Remove remaining formatting tags (align, text-align, style, etc.)
    text = re.sub(r'\b(align|style|coordinates|demonym)[^|\n]*', '', text)
    
    # Normalize or remove punctuation like bullet points
    text = re.sub(r'[•]', '', text)  # Remove bullet points

    # Ensure punctuation is properly spaced from both sides of words
    text = re.sub(r'([.,!?;:])(?!\s)', r' \1 ', text)  # Handle punctuation followed by a letter
    text = re.sub(r'(?<!\s)([.,!?;:])', r' \1 ', text)  # Handle punctuation preceded by a letter

    # Add spaces around parentheses
    text = re.sub(r'([()])', r' \1 ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def parse_wikipedia_xml(include_titles: bool = False, limit_articles: int|None = None):
    xml_file_path = get_default_xml_path()

    # Register the namespace
    ET.register_namespace('', "http://www.mediawiki.org/xml/export-0.11/")
    
    context = ET.iterparse(xml_file_path, events=('end',))
    articles = []

    for event, elem in context:
        if elem.tag.endswith('page'):
            # Use the full namespace in find
            title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}title')
            text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}text')
            
            if title_elem is not None and text_elem is not None:
                title = title_elem.text
                text = text_elem.text
                if title and text:
                    cleaned_text = clean_text(text)
                    articles.append((title, cleaned_text))
            elem.clear()
            if limit_articles and len(articles) == limit_articles:
                break

    return articles if include_titles else [a[1] for a in articles]

def get_default_xml_path():
    # Determine the absolute path to simplewiki.xml relative to this script
    script_dir = os.path.dirname(__file__)  # Path to data directory
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Path to deep-learning
    xml_file_path = os.path.join(project_root, 'data', 'simplewiki.xml')
    return xml_file_path

if __name__ == "__main__":
    articles = parse_wikipedia_xml()
    print(f"Extracted {len(articles)} articles")
    if articles:
        print("First article:")
        print(f"Title: {articles[0][0]}")
        print(f"Content: {articles[0][1][:200]}...")  # Print first 200 characters
    else:
        print("No articles were extracted. Check if the XML file is correctly formatted and not empty.")
