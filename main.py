import argparse
import os

from dotenv import load_dotenv
from nltk.chat.util import Chat, reflections

from chatml.mychat import MyChat

load_dotenv("config/secrets.env")


class CustomChat(Chat):
    def __init__(self, pairs, reflections, conversation_file):
        super().__init__(pairs, reflections)
        self.conversation_file = conversation_file
        self.chat = MyChat()

    def respond(self, message):
        with open(f"logs/{self.conversation_file}", "a") as f:
            f.write("User: " + message + "\n")

        response = self.chat.ask(message)

        with open(f"logs/{self.conversation_file}", "a") as f:
            f.write("Chatbot: " + response + "\n")

        return response

    def converse(self, quit="q"):
        user_input = ""
        print("Type your message to start the conversation or 'quit' to exit.")
        with open(f"logs/{self.conversation_file}", "w") as f:
            while user_input.lower() != quit:
                user_input = input("You: ")
                response = self.respond(user_input)
                print("Chatbot: " + response)
                f.write("User: " + user_input + "\n")
                f.write("Chatbot: " + response + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot CLI")
    parser.add_argument(
        "--conversation-file",
        type=str,
        default=os.getenv("CONVERSATION_FILE", "conversation_history.txt"),
        help="file to save conversation history to",
    )
    args = parser.parse_args()

    pairs = [
        [r"hi|hello|hey", ["Hello!", "Hi there!"]],
        [r"what is your name?", ["My name is Chatbot, nice to meet you!"]],
        [r"how are you?", ["I'm doing well, thanks for asking. How about you?"]],
        [r"i'm (.*)", ["Nice to hear that you are %1!"]],
        [
            r"what can you do?",
            ["I can answer simple questions, have conversations, and tell jokes!"],
        ],
        [
            r"tell me a joke",
            ["Why did the tomato turn red? Because it saw the salad dressing!"],
        ],
        [r"bye|goodbye", ["Goodbye!", "See you later!"]],
    ]

    custom_chatbot = CustomChat(pairs, reflections, args.conversation_file)
    custom_chatbot.converse()
