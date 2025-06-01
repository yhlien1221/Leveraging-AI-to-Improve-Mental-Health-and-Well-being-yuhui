import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model & Tokenizer from Hugging Face
model_name = "yhlien1221/dialoggpt-finetune-counsellor-data"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("🧠 Mental Health Counselor Chatbot")
st.caption("🤖 A fine-tuned DialoGPT model for supportive conversation.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Chat input
if prompt := st.chat_input("You:"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build full prompt from history
    full_prompt = ""
    for role, message in st.session_state.chat_history:
        if role == "user":
            full_prompt += f"User: {message} <|sep|> "
        elif role == "bot":
            full_prompt += f"Bot: {message} <|sep|> "
    full_prompt += f"User: {prompt} <|sep|> Bot:"

    # Tokenize and generate
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    output_ids = model.generate(
        input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    # Decode and extract response
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = decoded.split("<|sep|> Bot:")[-1].strip()

    # Show and store bot reply
    with st.chat_message("bot"):
        st.markdown(reply)

    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("bot", reply))
