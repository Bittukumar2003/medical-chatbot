from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables from the .env file.
load_dotenv()

# Retrieve the Pinecone API key from the environment configuration.
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Extract text data from the PDF files located in the data directory.
extracted_data = load_pdf("data/")

# Divide the extracted text into smaller, manageable chunks for processing.
text_chunks = text_split(extracted_data)

# Download and initialize the Hugging Face embedding model.
embeddings = download_hugging_face_embeddings()

# Initialize the Pinecone vector store and upload the text chunks as searchable embeddings.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name="medical-bot"
)