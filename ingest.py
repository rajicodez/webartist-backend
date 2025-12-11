import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone

# Load API Keys
load_dotenv()

# 1. SETUP API KEYS
# Ensure these are in your .env file or set in your environment
cohere_key = os.getenv("COHERE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# 2. INITIALIZE EMBEDDINGS (The "Translator")
# We use 'embed-english-v3.0' which creates 1024-dimension vectors
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# 3. LOAD YOUR DATA
# Create a file named 'hospital_data.txt' with all your hospital info
loader = TextLoader("webartist_data.txt", encoding="utf-8")
documents = loader.load()

# 4. SPLIT DATA (Because AI can't read a whole book at once)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(f"Split data into {len(docs)} chunks. Uploading to Pinecone...")

# 5. UPLOAD TO PINECONE
index_name = "webartist" # Make sure this matches your Pinecone Index Name

PineconeVectorStore.from_documents(
    docs, 
    index_name=index_name, 
    embedding=embeddings
)

print("✅ Success! Data uploaded to Pinecone.")