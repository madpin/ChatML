import copy
import datetime
import logging
import os

import openai
import pandas as pd
import tiktoken
import yaml


class MyChat:
    def __init__(self, yml_file: str = "base.yml", *, log_to_screen=True):
        yml_path = os.path.join(".", "config", yml_file)

        with open(yml_path, "r") as yml_file:
            self.config = yaml.safe_load(yml_file)

        if self.config.get("openai_api_key"):
            openai.api_key = self.config["openai_api_key"]
        elif self.config.get("openai_api_key_env"):
            openai.api_key = os.getenv(self.config["openai_api_key_env"])
        else:
            raise Exception("There is no API Key Setup")

        self.logger = logging.getLogger(f"MyChat_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            file_handler = logging.FileHandler("mychat.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            if log_to_screen:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        self.model_id = self.config["model"]

        self.encoding_name = tiktoken.encoding_for_model(self.model_id).name
        self.encoding = tiktoken.get_encoding(str(self.encoding_name))

        self.last_messages = None
        self.last_completion = None
        self.last_total_tokens = None
        self.last_price = None
        self.last_answer = None
        self.logger.debug("I'm Alive!")
        self.logger.debug(f"openai.api_key: {openai.api_key}")
        self.logger.debug(f"""self.config["openai_api_key_env"]: {self.config["openai_api_key_env"]}""")
    def get_prompts(self):
        return list(self.config["prompts"].keys())

    def get_date_str(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def to_file(self, question, answer):
        with open("logs/questions_yml.log", "a") as file:
            # Write the date to the file
            file.write(f"{self.get_date_str()}\n")
            # Write the first text block to the file
            file.write(f"{question}\n")
            # Write a separator between the two text blocks
            file.write("===\n")
            # Write the second text block to the file
            file.write(f"{answer}\n")

            file.write(
                f"Tokens: {self.last_total_tokens} | Price: ${self.last_price}\n"
            )
            file.write("=" * 60 + "\n")

    def get_messages(self, message, context=None, prompt=None):
        if prompt is None:
            if context is not None:
                return [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message},
                ]
            if isinstance(message, str):
                return [{"role": "user", "content": message}]

        if prompt not in self.config["prompts"]:
            raise Exception(f"This Prompt {prompt} doesn't exist in the Yml")

        prompts = copy.deepcopy(self.config["prompts"])

        prompt = prompts[prompt]
        if context:
            prompt.append({"role": "system", "content": context})
        if message:
            prompt.append({"role": "user", "content": message})

        return prompt

    def token_count(self, message, context=None, prompt=None):
        message = self.get_messages(message, context, prompt)
        return len(self.encoding.encode(str(message)))

    def ask(self, question, context=None, prompt=None):
        messages = self.get_messages(question, context, prompt)
        return self.completion(messages)

    def append_to_csv(self, message, prompt, tokens, file_path_csv="./logs/current_conv.csv"):
        row = {
            "class_id": id(self),
            "message": message,
            "prompt": prompt,
            "tokens": tokens,
        }
        self.logger.debug(f"file_path_csv: {file_path_csv}")
        try:
            df = pd.read_csv(file_path_csv)
        except Exception:
            df = pd.DataFrame()
        df = df.append(row, ignore_index=True)
        df.to_csv(file_path_csv, index=False)
        return df

    def completion(self, messages):
        self.last_messages = messages
        completion = openai.ChatCompletion.create(
            model=self.model_id, messages=messages
        )
        self.last_completion = completion
        self.last_total_tokens = completion["usage"]["total_tokens"]
        self.last_price = self.last_total_tokens * 0.000002
        answer = completion.choices[0].message.content
        self.last_answer = answer
        self.to_file(messages, answer)
        self.append_to_csv(completion, "prompt", self.last_total_tokens)
        return answer


# mc = MyChat()
