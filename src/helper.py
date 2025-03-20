import os
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Check if MPS (Metal Performance Shaders) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
    return splitter.split_text(text)

# Function to create a vector store using Hugging Face embeddings
def get_vector_store(chunks):
    # Initialize Hugging Face embeddings model (sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS vector store from text chunks and embeddings
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    return vector_store

# Function to create a conversational chain using Hugging Face chat models
def get_conversational_chain(vector_store):
    # Initialize Hugging Face pipeline with MPS support for text generation
    hf_pipeline = pipeline(
        task="text-generation",
        model="google/flan-t5-small",  # Use a smaller model for faster inference
        device=0 if device == "mps" else -1,  # Use MPS if available, otherwise CPU
        max_length=256,
        temperature=0.7,
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    
    return chain

