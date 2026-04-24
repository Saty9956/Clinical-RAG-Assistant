import os
from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 1. Initialize Gemini API (Free Tier: 250k TPM)
load_dotenv()

print("Initializing Gemini LLM and Local Embeddings...")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview", # Reverted to the stable API model string
    temperature=0.0,
)

# 2. Local Embeddings (Must match the model used in ingest_data.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Connect to the Persistent Vector Database
print("Loading persistent FAISS vector database from disk...")
vectorstore = FAISS.load_local(
    "faiss_medical_index", 
    embeddings, 
    allow_dangerous_deserialization=True 
)

# ADD THIS LINE TO VERIFY DATA
print(f"✅ SUCCESSFULLY LOADED {vectorstore.index.ntotal} VECTORS.")

# We pass the top 4 chunks to Gemini
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
# 4. The Clinical Guardrail Prompt
# NEW IMPROVED PROMPT
template = """You are a professional Medical Assistant. 
Use the following pieces of clinical context to answer the user's question. 
If the context doesn't contain the answer, say you don't know based on the data, but try your best to be helpful.

CONTEXT:
{context}

USER QUESTION: {question}

CLINICAL ANSWER:"""

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# RECONFIGURED CHAIN
# We add return_source_documents=True so we can see what was used
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" means "stuff all 4 documents into the prompt"
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def get_medical_answer(query):
    try:
        # Run the chain
        result = qa_chain.invoke({"query": query})
        
        ai_response = result["result"]
        # Extract the source text properly
        sources = [doc.page_content for doc in result["source_documents"]]
        
        return ai_response, sources
    except Exception as e:
        print(f"Error in RAG Engine: {e}")
        return "I encountered an error while searching clinical databases.", []