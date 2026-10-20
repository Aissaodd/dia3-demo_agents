import streamlit as st
from llm_agent import ConversationAgent


class UserInterface:
    def __init__(self):
        st.set_page_config(page_title="Jarvis")
        st.title("Barbe-Quasar")
        self.conversation_agent_object = self.get_persistant_conversation_agent()

    def get_persistant_conversation_agent(self):
        if "conversation_agent_object" not in st.session_state:
            st.session_state.conversation_agent_object = ConversationAgent()
        return st.session_state.conversation_agent_object

    def show_discussion_history(self):
        for message in self.conversation_agent_object.get_conversation_history():
            with st.chat_message(message["role"]):
                content = message["content"]
                if isinstance(content, str):
                    st.markdown(content)
                else: 
                    for part in content:
                        if part["type"] == "text":
                            st.markdown(part["text"])
                        else:
                            st.image(part["image_url"]["url"], width=300)

    def run(self):
        prompt = st.chat_input(
            "Que veux tu moussaillon ?",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
            accept_audio=True,
        )

        if prompt:
            text = prompt.text or ""

            if prompt.audio:
                with st.spinner("Transcription..."):
                    try:
                        transcription = self.conversation_agent_object.transcribe_audio(
                            prompt.audio
                        )
                    except Exception as e:
                        st.error(f"Erreur de transcription : {e}")
                        transcription = ""
                text = f"{text} {transcription}".strip()

            if text or prompt.files:
                with st.spinner("Jarvis réfléchit..."):
                    try:
                        self.conversation_agent_object.ask_llm(
                            text, images=prompt.files
                        )
                    except Exception as e:
                        st.error(f"Erreur Groq : {e}")

        self.show_discussion_history()


if __name__ == "__main__":
    user_interface_object = UserInterface()
    user_interface_object.run()




