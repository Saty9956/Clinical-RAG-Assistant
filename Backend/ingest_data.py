from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import time

print("🚀 Starting Full-Scale Clinical Ingestion...")

# 1. Download the full MedQuAD Dataset
print("📥 Downloading full MedQuAD dataset from HuggingFace...")
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

# 2. Format clinical data (Processing all 47,000+ rows)
print(f"📄 Formatting {len(dataset)} clinical records...")
clinical_texts = []
for i in tqdm(range(len(dataset)), desc="Formatting"):
    q = dataset[i]['Question']
    a = dataset[i]['Answer']
    clinical_texts.append(f"Question: {q}\nClinical Answer: {a}")

# 3. Intelligent Chunking
# We use a slightly larger chunk size for better clinical context retention
print("✂️ Chunking documents for vectorization...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = text_splitter.create_documents(clinical_texts)
print(f"Created {len(docs)} document chunks.")

# 4. Initialize Local Embedding Model
# This runs on your CPU. It's the most compute-heavy part.
print("🧠 Initializing Local Embedding Model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Embed and Index (The Heavy Lifting)
print(f"⚡ Vectorizing {len(docs)} chunks. This will take 30-60 minutes on most CPUs...")
start_time = time.time()

# FAISS.from_documents can be slow for 47k rows. 
# We process in batches of 1000 to prevent RAM spikes.
batch_size = 1000
vectorstore = None

for i in tqdm(range(0, len(docs), batch_size), desc="Embedding Batches"):
    batch = docs[i : i + batch_size]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        # Add subsequent batches to the existing index
        temp_store = FAISS.from_documents(batch, embeddings)
        vectorstore.merge_from(temp_store)

end_time = time.time()
print(f"✨ Vectorization complete! Total time: {(end_time - start_time)/60:.2f} minutes.")

# 6. Save the Full Database
print("💾 Saving massive medical index to 'faiss_medical_index'...")
vectorstore.save_local("faiss_medical_index")

print("\n✅ MISSION ACCOMPLISHED: You now have a full-scale clinical brain on your disk.")