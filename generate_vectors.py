import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Scrape documentation
def scrape_docs(base_url):
    parsed_base = urlparse(base_url) # Parse the base URL to extract the base domain and path
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"  # Create base domain
    base_path = parsed_base.path.rstrip("/")  # Remove trailing /

    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to fetch the base URL")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select("a[href]")  # Find all anchor tags with an href attribute
    docs = []
    visited = set()

    for link in links:
        href = link['href']


        if not href or href.startswith("#") or "javascript:" in href:
            continue

        full_url = href

        if full_url in visited:
            continue

        visited.add(full_url) # Add to visited to prevent duplication

        # Check if the link path starts with the base path or base url
        if href.startswith(base_path):
            # Combine base domain with the link's path
            full_url = f"{base_domain}{href}"

        # Skip URLs that don't match the base path
        if not full_url.startswith(base_url):
            continue

        # Fetch the page and extract content
        try:
            page_response = requests.get(full_url, timeout=10)  # Add a timeout
            page_response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.RequestException as e:
            print(f"Failed to fetch {full_url}: {e}")
            continue

        if page_response.status_code == 200:
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            content = page_soup.select_one("main").get_text() # Check this can be refined so we don't save copies of same data
            docs.append({"url": full_url, "title": link.text.strip() or "Untitled", "content": content})

    return docs


# Step 2: Preprocess content
base_url = "https://docs.github.com/en"
docs = scrape_docs(base_url)

# Combine content and keep metadata
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = []
metadata = []

for doc in docs:
    chunks = text_splitter.split_text(doc['content'])
    for chunk in chunks:
        documents.append(chunk)
        metadata.append({"url": doc['url'], "title": doc['title']})

# Step 3: Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create a vector store
vector_store = FAISS.from_texts(documents, embedding=embeddings, metadatas=metadata)
vector_store.save_local("docs_vectorstore")
print("Vector store created successfully!")

