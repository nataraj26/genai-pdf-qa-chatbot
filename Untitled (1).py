#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Load OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']

def load_pdf_to_db(file_path):
    # Load the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # Set retriever to fetch relevant document chunks
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
def create_conversational_chain(retriever):
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define the conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),  # Using OpenAI Chat model
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
if __name__ == "__main__":
    # Load the PDF file
    pdf_file_path = "Unit I Introduction.pdf"  # Replace with your file path
    retriever = load_pdf_to_db(pdf_file_path)
    
    # Create chatbot chain
    chatbot = create_conversational_chain(retriever)
    
    # Start a conversation
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        # NATARAJ KUMARAN S (212223230137)
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])

