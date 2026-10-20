import base64
import os

from dotenv import load_dotenv
from groq import Groq

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"


class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
        self.conversation_history = [
            {
                "role": "system",
                "content": self.read_file(file_path="./context.txt"),
            }
        ]

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def image_to_data_url(image_file):
        b64 = base64.b64encode(image_file.getvalue()).decode("utf-8")
        return f"data:{image_file.type};base64,{b64}"

    def get_conversation_history(self):
        return self.conversation_history[1:]

    def update_conversation_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def _history_has_images(self):
        return any(isinstance(m["content"], list) for m in self.conversation_history)

    def ask_llm(self, text_interaction, images=None):
        if images:
            content = [
                {"type": "text", "text": text_interaction or "Décris cette image."}
            ]
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self.image_to_data_url(image)},
                    }
                )
        else:
            content = text_interaction

        self.update_conversation_history(role="user", content=content)

        if self._history_has_images():
            kwargs = {
                "model": VISION_MODEL,
                "reasoning_format": "hidden",  # retire cette ligne si erreur 400
                "max_completion_tokens": 4096,
            }
        else:
            kwargs = {"model": TEXT_MODEL, "include_reasoning": False}

        try:
            completion = self.groq_client.chat.completions.create(
                messages=self.conversation_history,
                **kwargs,
            )
        except Exception:
            self.conversation_history.pop()
            raise

        response = (completion.choices[0].message.content or "").strip()
        if not response:
            response = "(Le modèle n'a renvoyé aucune réponse, réessaie.)"

        self.update_conversation_history(role="assistant", content=response)
        return response


if __name__ == "__main__":
    agent = ConversationAgent()
    question = input("Pose ta question :\n")
    print("\n")
    print(agent.ask_llm(question))