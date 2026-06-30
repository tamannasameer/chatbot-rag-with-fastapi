import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Shared global variables for the RAG chain
rag_chain = None
DB_DIR = "chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rebuild_rag_chain():
    """Helper function to load/reload the RAG chain from the Chroma directory."""
    global rag_chain
    
    if not os.path.exists(DB_DIR):
        return None

    # Connect to the database
    vector_store = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful and friendly assistant. 

- If the user says hello, greets you, or asks general pleasantries, respond warmly and naturally.
- For factual or technical questions, look at the provided retrieved context below.
- If the question requires specific knowledge that is NOT found in the context and isn't a general greeting/conversation, politely say: "I cannot find that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:
""")

    # Build Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# Modern FastAPI Lifespan instead of app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Try to load the chain at startup if a database already exists."""
    rebuild_rag_chain()
    if rag_chain:
        print("Existing Chroma database loaded successfully.")
    else:
        print("No existing database found. System waiting for a PDF upload.")
    yield


app = FastAPI(
    title="Dynamic RAG Chatbot API",
    description="Upload a PDF and chat with it instantly",
    version="2.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

# Enable Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin interface access. Narrow this down for production builds.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str


# --- Endpoints ---

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(..., description="Upload your PDF file here")):
    global rag_chain
    
    print(f"\n[INFO] Received upload request for file: {file.filename}")
    
    # 1. Validation: Ensure it's a PDF
    if not file.filename.lower().endswith('.pdf'):
        print("[ERROR] Rejected non-PDF file.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf_path = f"temp_{file.filename}"
    print(f"[INFO] Creating temporary file at: {temp_pdf_path}")
    
    try:
        # 2. Save file using chunks safely
        print("[INFO] Reading and saving uploaded file content...")
        with open(temp_pdf_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # Read 1MB chunks
                buffer.write(content)
        
        print("[INFO] PDF saved locally. Starting PyPDFLoader...")
        # 3. Process the PDF (Ingestion logic)
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        print(f"[INFO] Loaded {len(documents)} pages. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"[INFO] Split into {len(docs)} chunks.")
        
        # 4. Save directly into ChromaDB
        print("[INFO] Writing vectors to ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        
        # 5. Reinitialize the RAG chain
        print("[INFO] Rebuilding RAG chain...")
        rebuild_rag_chain()
        print("[SUCCESS] Processing complete!")
        
        return {
            "status": "Success", 
            "message": f"Successfully processed '{file.filename}' and added {len(docs)} chunks."
        }
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        
    finally:
        # Clean up the temporary file safely
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"[INFO] Cleaned up temporary file: {temp_pdf_path}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    global rag_chain
    
    # If a user tries to chat before uploading anything
    if rag_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No knowledge base found. Please upload a PDF file via /upload first."
        )
    
    try:
        user_query = payload.question.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
            
        response = rag_chain.invoke(user_query)
        return ChatResponse(answer=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")