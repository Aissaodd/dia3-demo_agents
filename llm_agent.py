from openai import OpenAI

from dotenv import load_dotenv
import os


class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_KEY"], base_url="https://api.deepseek.com")
        self.conversation_history = [
                {
                    "role": "system",
                    "content": self.read_file(file_path="./context.txt")
                }
            ]


    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()


    def get_conversation_history(self):
        return self.conversation_history[1: ]


    def update_conversation_history(self, role, content):
        self.conversation_history.append(
            {
            "role": role,
            "content": content,
            }
        )


    def ask_llm(self, text_interaction):

        self.update_conversation_history(role="user", content=text_interaction)

        response = self.client.chat.completions.create(
            messages=self.conversation_history,
            model="deepseek-chat",
        ).choices[0].message.content


        self.update_conversation_history(role="assistant", content=response)





























if __name__ == "__main__":
    question = input("Pose ta question :\n")
    response = ask_llm(question)
    print("\n")
    print(response)