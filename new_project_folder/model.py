# app.py - Bilingual AgriChat Bot (Streamlit + Rasa)
# Streamlit = UI only | Rasa = Brain | RAG handled inside Rasa

import streamlit as st
import requests

# --------------------- CONFIG ---------------------
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

# --------------------- STREAMLIT UI ---------------------
st.set_page_config(
    page_title="Bilingual Farming Advisory Chatbot",
    page_icon="🌱",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #121212; }
    .stApp { background: linear-gradient(135deg, #1c1c1c, #121212); }
    .chat-user {
        background-color: #333333;
        border-radius: 15px;
        padding: 10px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        color: #ffffff;
    }
    .chat-bot {
        background-color: #444444;
        border-radius: 15px;
        padding: 10px;
        margin:10px 0;
        max-width: 80%;
        align-self: flex-start;
        color: #ffffff;
    }
    h1 { color: #ffffff; text-align: center; }
    body, p, div, span { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Bilingual Farming Chatbot")

# --------------------- SESSION STATE ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I’m your farming assistant. Ask me anything in English or Pidgin 😊"
        }
    ]

# --------------------- DISPLAY CHAT ---------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user"><strong>You:</strong> {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bot"><strong>Bot:</strong> {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# --------------------- INPUT ---------------------
user_input = st.chat_input("Type your farming question here...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.markdown(
        f'<div class="chat-user"><strong>You:</strong> {user_input}</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Thinking..."):
        try:
            rasa_resp = requests.post(
                RASA_URL,
                json={"sender": "user", "message": user_input},
                timeout=10
            ).json()
        except Exception:
            rasa_resp = []

        final_answer = None
        source = "Answered by Rasa"

        if isinstance(rasa_resp, list):
            texts = []
            for m in rasa_resp:
                if isinstance(m, dict) and "text" in m:
                    texts.append(m["text"])

            if texts:
                final_answer = "\n".join(texts)

        elif isinstance(rasa_resp, dict):
            final_answer = rasa_resp.get("text")

        elif isinstance(rasa_resp, str):
            final_answer = rasa_resp


            # Optional: detect RAG answers sent by Rasa
            if any("I don’t have this in my training data" in t for t in texts):
                source = "Answered by RAG (via Rasa)"

        if not final_answer:
            final_answer = "I’m having trouble answering right now. Please try again."
            source = "System fallback"
        st.write("DEBUG RASA RESPONSE:", rasa_resp)
    # Show bot response
    st.markdown(
        f'<div class="chat-bot"><strong>Bot:</strong> {final_answer}'
        f'<br><small><i>{source}</i></small></div>',
        unsafe_allow_html=True
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )

st.markdown("---")


