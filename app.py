import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    """Formats retrieved document chunks into a clean text block for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("=== Initializing Chatbot System (ChromaDB) ===")

    # 1. Verify and load the locally saved Chroma database
    if not os.path.exists("chroma_db"):
        print("Error: 'chroma_db' folder not found! Run 'python ingest.py' first.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Connect back to the saved directory
    vector_store = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Chroma database loaded successfully.")

    # 2. Initialize the Language Model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. Craft the RAG prompt
    # 3. Craft the RAG prompt
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

    # 4. Build the LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 5. Execution Loop
    print("\nChatbot is ready! Ask anything about your PDF data. (Type 'exit' to quit)\n")
    
    while True:
        try:
            user_query = input("You: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            # Process query through the RAG chain
            response = rag_chain.invoke(user_query)
            print(f"\nAI: {response}\n" + "-"*40 + "\n")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()