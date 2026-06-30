import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()



def main():
    print("=== Starting PDF Ingestion (ChromaDB) ===")
    
    # 1. Point to your local PDF file
    pdf_filename = "ai.pdf" 
    
    if not os.path.exists(pdf_filename):
        print(f"Error: '{pdf_filename}' not found! Please place your PDF in this folder.")
        return

    # 2. Load and parse the PDF
    print(f"Loading and parsing {pdf_filename}...")
    loader = PyPDFLoader(pdf_filename)
    documents = loader.load()

    # 3. Split the text into manageable chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks from your PDF.")

    # 4. Generate embeddings and save directly to ChromaDB
    print("Generating embeddings and building Chroma database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chroma automatically persists/saves data to the directory specified
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="chroma_db"
    )
    
    print("Success! Your database is built and saved to the 'chroma_db' folder.\n")

if __name__ == "__main__":
    main()