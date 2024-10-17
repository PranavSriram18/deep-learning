import requests
import bz2
import os

def download_wikipedia_simple():
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    local_filename = "simplewiki.xml.bz2"
    
    print("Downloading Wikipedia Simple English dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("Download complete. Extracting...")
    with bz2.BZ2File(local_filename) as compressed, open("simplewiki.xml", 'wb') as uncompressed:
        for data in iter(lambda: compressed.read(100 * 1024), b''):
            uncompressed.write(data)
    
    print("Extraction complete. Cleaning up...")
    os.remove(local_filename)
    
    print("Process finished. The extracted file is 'simplewiki.xml'")

if __name__ == "__main__":
    download_wikipedia_simple()