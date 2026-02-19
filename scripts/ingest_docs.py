import sys
sys.path = [p for p in sys.path if "Object_Detection" not in p]
import sentence_transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
import os

# 1. Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "documents")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Created directory: {DATA_DIR}. Please place your submarine PDFs there.")

OUTPUT_DB = os.path.join(PROJECT_ROOT, "models", "vector_indices", "core_index")

# 2. Load Documents
print(f"Scanning {DATA_DIR} for PDFs and root for .txt files...")
documents = []

import pypdf
from langchain_core.documents import Document

# Load from documents folder (PDFs), searching recursively
if os.path.exists(DATA_DIR):
    print(f"Scanning {DATA_DIR} recursively for PDFs...")
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.endswith(".pdf"):
                full_path = os.path.join(root, filename)
                print(f"Indexing PDF: {filename}...")
                try:
                    reader = pypdf.PdfReader(full_path)
                    print(f"  - Reading {len(reader.pages)} pages...")
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i+1}
                            ))
                    print(f"  - Successfully extracted text.")
                except Exception as e:
                    print(f"  - Error loading {filename}: {e}")

# Load local txt manuals
MANUALS_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "manuals")
for filename in ["manual_physics.txt", "manual_engineering.txt", "manual_acoustic.txt", "manual_mdfls.txt", "manual_yolosonar.txt", "manual_metrics.txt"]:
    file_path = os.path.join(MANUALS_DIR, filename)
    if os.path.exists(file_path):
        print(f"Indexing Text: {filename}...")
        loader = TextLoader(file_path)
        documents.extend(loader.load())

if not documents:
    print("Warning: No documents found to index.")
    sys.exit()

print(f"Total pages/documents loaded: {len(documents)}")

# 3. Split Text
# Reduced chunk size for higher resolution search, especially for names/citations
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
texts = text_splitter.split_documents(documents)

# 4. Create Embeddings & Store
print(f"Processing {len(texts)} total text chunks...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building/Updating Vector Store (FAISS)... This may take a few minutes for 50MB+ PDFs...")
db = FAISS.from_documents(texts, embeddings)
db.save_local(OUTPUT_DB)

print(f"Success! Knowledge DB updated with latest datasets.")


