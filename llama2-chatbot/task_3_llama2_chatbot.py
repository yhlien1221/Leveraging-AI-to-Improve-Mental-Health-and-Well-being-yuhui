# app.py
import streamlit as st
import os
import requests
from ctransformers import AutoModelForCausalLM

# Model metadata
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
MODEL_FILE = "tinyllama-1.1b-chat.Q4_0.gguf"

# 1) Load model using ctransformers
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("🔄 Downloading TinyLlama model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_FILE, "wb") as f:
                f.write(response.content)

    return AutoModelForCausalLM.from_pretrained(
        model_path=MODEL_FILE,
        model_type="llama"
    )

llm = load_model()

# 2) Streamlit UI
st.title("🧠 Mental Health Counselor Chatbot")
st.caption("💬 Powered by TinyLlama + ctransformers")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# New input
if user_input := st.chat_input("You:"):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # Rebuild prompt
    MAX_HISTORY_TURNS = 4
    hist = st.session_state.chat_history[-MAX_HISTORY_TURNS*2:]
    prompt = ""
    for role, msg in hist:
        prompt += f"{'User' if role == 'user' else 'Bot'}: {msg}\n"
    prompt += "Bot:"

    # Generate reply
    with st.spinner("🤖 Thinking..."):
        reply = llm(prompt, max_new_tokens=150)
    reply = reply.split("User:")[0].strip()

    st.chat_message("assistant").markdown(reply)
    st.session_state.chat_history.append(("assistant", reply))
