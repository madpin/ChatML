from dotenv import load_dotenv

from chatml.mychat import MyChat

load_dotenv("config/secrets.env")

chat = MyChat()
qs = ["Quem descobriu o brasil?"]
for q in qs:
    print(f"Ask: {q}")
    ans = chat.ask(q)
    print(f"Ans: {ans}")
