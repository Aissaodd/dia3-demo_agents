from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def ask_llm(question, history=None):
    client = Groq(api_key=os.environ["GROQ_KEY"])

    messages = [
        {"role": "system", "content": read_file(file_path="./context.txt")}
    ]
    if history:
        messages.extend(history)  # anciens échanges
    messages.append({"role": "user", "content": question})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        include_reasoning=False,
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    history = []
    while True:
        question = input("Pose ta question (ou 'q' pour quitter) :\n")
        if question.lower() == "q":
            break
        response = ask_llm(question, history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        print("\n" + response + "\n")