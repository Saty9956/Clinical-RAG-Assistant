from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from rag_engine import get_medical_answer # Imports your File 1 logic

# Initialize the API
app = FastAPI(title="Medical AI RAG Backend", version="1.0")

# Define the data structure we expect from the frontend
class ChatRequest(BaseModel):
    session_id: str
    user_message: str

# Define the data structure we will send back to the frontend
class ChatResponse(BaseModel):
    ai_response: str
    sources_used: list

@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """
    This endpoint receives a message, runs safety triage, 
    and returns the Gemini AI response.
    """
    user_query = request.user_message.lower()

    # 1. HARD GUARDRAIL: Emergency Triage Routing (Zero-Latency)
    # Production medical apps intercept emergencies before hitting the LLM
    emergency_keywords = ["heart attack", "stroke", "suicide", "chest pain", "emergency"]
    if any(keyword in user_query for keyword in emergency_keywords):
        return ChatResponse(
            ai_response="🚨 **EMERGENCY WARNING:** Please seek immediate medical attention or call emergency services. I am an AI, not a doctor.",
            sources_used=["FastAPI Triage Router"]
        )

    # 2. RAG Execution (Using Gemini & FAISS)
    try:
        # Pass the text to File 1
        ai_answer, sources = get_medical_answer(request.user_message)

        return ChatResponse(
            ai_response=ai_answer,
            sources_used=sources
        )
    except Exception as e:
        print(f"Backend API Error: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response.")

# This allows you to run the server locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)