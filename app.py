from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# 1. Setup Pinecone API Key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 2. Load Embeddings
embeddings = download_hugging_face_embeddings()

# 3. Initialize Pinecone and VectorStore
index_name = "medical-chatbot"

# Load the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)

# 4. Setup the Retriever
# Note: This was missing in your original code
retriever = docsearch.as_retriever(search_kwargs={'k': 2})

# 5. Define the Prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# 6. Initialize the LLM
# Ensure this path is correct for your local machine
model_path = r"C:\Users\HP\Downloads\End-to-end-Medical-Chatbot-using-Llama2\model\llama-2-7b-chat.ggmlv3.q4_0 (1).bin"

llm = CTransformers(
    model=model_path,
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# 7. Setup the QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # Correct way to invoke the chain
    result = qa.invoke({"query": msg})
    
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)