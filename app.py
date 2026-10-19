import streamlit as st 
from llm_agent import ask_llm


st.set_page_config(page_title="Jarvis")
st.title("Barbe-Quasar")


user_input = st.chat_input(
	"Que veux tu moussaillon ?",
	accept_file=False,
	accept_audio=False
	)

if user_input :
	response = ask_llm(user_input)
	
	with st.chat_message("user"):
		st.markdown(user_input)

	with st.chat_message("system"):
		st.markdown(response)


