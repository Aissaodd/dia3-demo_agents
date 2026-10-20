import streamlit as st
import llm_agent
from llm_agent import ask_llm

st.set_page_config(page_title="Jarvis")
st.title("Barbe-Quasar")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Que veux tu moussaillon ?",
    accept_file=False,
    accept_audio=False,
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            response = ask_llm(user_input, history=st.session_state.messages)
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})