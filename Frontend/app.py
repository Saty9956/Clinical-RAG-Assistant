import streamlit as st
import requests

# Set up the UI page aesthetics
st.set_page_config(page_title="Clinical AI Assistant", page_icon="🩺", layout="centered")
st.title("🩺 Clinical RAG Assistant")
st.caption("A production-grade medical AI pipeline using local retrieval (FAISS) and Gemini 3.1 Flash-Lite.")

# The URL where your FastAPI backend is running
API_URL = "http://127.0.0.1:8000/chat"

# Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
    
    # 1. Display User Message in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Ping the FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical databases..."):
            try:
                # Send the payload to the backend
                payload = {"session_id": "user_123", "user_message": prompt}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    ai_text = data["ai_response"]
                    sources = data["sources_used"]
                    
                    # Display the AI response
                    st.markdown(ai_text)
                    
                    # Display the sources nicely if they exist
                    if sources and "Triage Router" not in sources:
                        with st.expander("📚 View Clinical Sources Used"):
                            for idx, source in enumerate(sources):
                                st.write(f"**Source {idx+1}:** {source}")
                    elif "Triage Router" in sources:
                        st.caption("Source: Triage Router (Bypassed LLM for safety)")
                    
                    # Save AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                else:
                    st.error(f"Backend Error: Received status code {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to Backend API. Make sure FastAPI is running in another terminal window!")