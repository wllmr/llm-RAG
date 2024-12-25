import sys
import subprocess
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Disables multithreading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the query is provided via the command line
if len(sys.argv) < 2:
    print("Usage: python rag.py '<your query>'")
    sys.exit(1)

query = sys.argv[1]

# Load the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS vector store
try:
    vector_store = FAISS.load_local(
        "./docs_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print("Error loading FAISS vector store:", e)
    sys.exit(1)

# Perform similarity search to retrieve relevant context
try:
    results = vector_store.similarity_search(query, k=3)  # Retrieve top 3 matches
    context = " ".join([result.page_content for result in results])
except Exception as e:
    print("Error retrieving context:", e)
    sys.exit(1)

# Truncate context if needed
max_context_length = 1500  # Adjust based on your model's context window
context = context[:max_context_length]

# Construct the full prompt
full_prompt = f"""
You are a AI chatbot. Your intention is to assist in finding information and explaining it in depth. Be friendly and explain in depth how to do the actions.

Context: {context}

Question: {query}

Answer:
"""

# Use the Ollama CLI to generate an answer
try:
    result = subprocess.run(
        ["ollama", "run", "llama3-chatqa:8b"],
        input=full_prompt,
        text=True,
        capture_output=True,
        check=True
    )
    print(result.stdout.strip())
except subprocess.CalledProcessError as e:
    print("Error generating response:")
    print("Exit Code:", e.returncode)
    print("Standard Output:", e.stdout.strip())
    print("Error Output:", e.stderr.strip())
