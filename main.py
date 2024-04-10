from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get('COHERE_API_KEY')

# 1. Retriever
# Loader y carga de los documentos
loader = PyPDFLoader("data/document.pdf")
docs = loader.load()
# Crear una base de datos vectorial en memoria con la información de los documentos
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# Descargar los embbedings
embeddings = CohereEmbeddings(cohere_api_key=api_key)
# Crear el retriever
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# LLM
llm = ChatCohere(cohere_api_key=api_key)
qa_chain = load_qa_chain(llm, chain_type="stuff")

def get_llm_response(query, vector_db, chain, history):
    matching_docs = vector_db.similarity_search(query)
    answer = chain.invoke({"question": query, "chat_history" : history, "input_documents":matching_docs})
    return answer

history = []

# App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    
    response = get_llm_response(question, vector, qa_chain, history)
    
    return response['output_text']

if __name__ == '__main__':
    app.run(debug=True)