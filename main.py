from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()


def ask_llm(question):
    
    client = Groq(api_key=os.environ["GROQ_KEY"])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant sympa"
            },

            {
                "role": "user",
                "content": question,
            }
        ],

        model="qwen/qwen3.6-27b",
        include_reasoning=False,
    )


    response = chat_completion.choices[0].message.content
    return response


if __name__ == "__main__":
    question = input("Pose ta question :\n")
    response = ask_llm(question)
    print("\n")
    print(response)