import os
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    batch_size = 32
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors.extend(FAISS.from_texts(batch, embedding=embeddings))
    
    return FAISS(vectors)

def get_conversational_chain(vector_store):
    hf_pipeline = pipeline(
        task="text-generation",
        model="google/flan-t5-small",
        device=0 if device == "mps" else -1,  
        max_length=256,
        temperature=0.7,
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    
    return chain

