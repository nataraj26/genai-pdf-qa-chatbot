## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
The aim of this project is to develop an AI-powered conversational chatbot that can read, understand, and answer questions from PDF documents. By leveraging LangChain, OpenAI embeddings, and conversational memory, the system enables users to interact with PDF content in a natural, question–answer format, making document navigation and knowledge extraction faster and easier.

### PROBLEM STATEMENT:

### STEP 1: Document Loading

Import the required libraries.
Load the PDF file using PyPDFLoader.
Extract text content from the PDF for further processing.

### STEP 2: Text Preprocessing and Embedding

Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.
Convert text chunks into vector embeddings using OpenAIEmbeddings.
Store the embeddings in a vector database (DocArrayInMemorySearch/FAISS/Chroma) for efficient retrieval.

### STEP 3: Conversational Chain Setup

Initialize ConversationBufferMemory to keep track of the conversation history.
Create a ConversationalRetrievalChain using ChatOpenAI as the LLM.
Connect the retriever with the conversational chain for contextual Q&A.

### STEP 4: User Interaction

Build an interactive loop to take user queries.
Retrieve relevant chunks from the database.Generate accurate answers using the chatbot.
Continue until the user exits the conversatio
### PROGRAM:
```python
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
    pdf_file_path = "212223040021-Ashwin Kumar A-Report.pdf"  # Replace with your file path
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
        
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])
```

### OUTPUT:
<img width="1379" height="644" alt="Screenshot 2025-09-26 115142" src="https://github.com/user-attachments/assets/8f6cf94e-3cb1-42a7-b0d0-ad7e196494bb" />


### RESULT:
The system is able to load PDF documents and split them into meaningful chunks.
Using OpenAI embeddings, the chatbot retrieves the most relevant information from the document.
The Conversational Retrieval Chain allows the chatbot to maintain conversation context and provide accurate answers.
The user can interact with the chatbot in a natural, question–answer format and efficiently extract knowledge from PDF files.
