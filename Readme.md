# 🩺 Clinical-RAG Assistant

A production-grade Medical AI pipeline utilizing **Retrieval-Augmented Generation (RAG)** to provide grounded, clinical answers from a local vector database. 

## 🏗️ System Architecture
The system is decoupled into a microservice architecture:
1. **Frontend (Streamlit):** Reactive UI for real-time clinician interaction.
2. **Backend (FastAPI):** High-performance API handling query orchestration and safety routing.
3. **Engine (LangChain/FAISS):** Local vector store containing **46,000+ NIH MedQuAD clinical records** for contextually relevant retrieval.



## ⚡ Key Technical Features
* **Zero-Latency Triage:** Custom asynchronous router intercepts emergency-related queries (e.g., "stroke", "heart attack") before model execution.
* **Persistent Vector Storage:** Local FAISS indexing allows for high-speed similarity search without recurring embedding costs.
* **Fact-Grounded Inference:** Leverages **Gemini 3.1 Flash-Lite** with temperature $0.0$ to ensure responses are derived strictly from retrieved NIH clinical data.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Orchestration:** LangChain & LangChain-Classic
- **Vector DB:** FAISS
- **Model:** Google Gemini 3.1 Flash-Lite (via Vertex AI/Google AI SDK)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **UI:** Streamlit

## 🔧 Setup Instructions
1. Clone the repo.
2. Create a `.env` file with your `GOOGLE_API_KEY`.
3. Run `pip install -r requirements.txt`.
4. Execute `python backend/ingest_data.py` to build the local index.
5. Launch the backend: `python backend/main.py`.
6. Launch the frontend: `streamlit run frontend/app.py`.