import copy
import datetime
import logging
import os

import openai
import pandas as pd
import tiktoken
import yaml


class MyChat:
    def __init__(self, yml_file: str = "base.yml", log_to_screen: bool = True):
        """
        Initialize a new MyChat instance.

        Args:
        - yml_file (str): The name of the YAML file containing configuration options.
        - log_to_screen (bool): Whether to log messages to the console in addition to a file.

        Raises:
        - Exception: If no OpenAI API key is set up in the configuration file.

        """
        self.set_logger(log_to_screen)

        # Load configuration options from YAML file.
        yml_path = os.path.join(".", "config", yml_file)
        with open(yml_path, "r") as yml_file:
            self.config = yaml.safe_load(yml_file)

        # Set up OpenAI API key.
        api_key = self.config.get("openai_api_key")
        api_key_env = self.config.get("openai_api_key_env")
        if api_key:
            openai.api_key = api_key
        elif api_key_env:
            openai.api_key = os.getenv(api_key_env)
        else:
            raise Exception("No OpenAI API key set up in configuration file.")

        # Set up other instance variables.
        self.model_id = self.config["model"]
        self.encoding_name = tiktoken.encoding_for_model(self.model_id).name
        self.encoding = tiktoken.get_encoding(str(self.encoding_name))
        self.last_messages = None
        self.last_completion = None
        self.last_total_tokens = None
        self.last_price = None
        self.last_answer = None
        self.logger.debug("MyChat instance created.")
        self.logger.debug(f"OpenAI API key: {openai.api_key}")
        self.logger.debug(f"API key environment variable: {api_key_env}")

    def set_logger(self, log_to_screen: bool):
        """
        Configures and returns a logger object.

        Args:
            log_to_screen: Whether to log messages to the console.

        Returns:
            A logger object configured with a file handler and, optionally, a console handler.
        """
        self.logger = logging.getLogger(f"MyChat_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            file_handler = logging.FileHandler(os.path.join("logs", "mychat.log"))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

            if log_to_screen:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        return self.logger

    def get_prompts(self):
        return list(self.config["prompts"].keys())

    def get_date_str(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def to_file(self, question: str, answer: str):
        """Writes the question and answer to a log file.

        Args:
            question (str): The user's question.
            answer (str): The chatbot's answer.

        Returns:
            None

        Raises:
            FileNotFoundError: If the log file cannot be found.

        """
        log_path = os.path.join("logs", "questions_yml.log")
        with open(log_path, "a") as log_file:
            log_file.write(f"{self.get_date_str()}\n")
            log_file.write(f"{question}\n")
            log_file.write("===\n")
            log_file.write(f"{answer}\n")
            log_file.write(
                f"Tokens: {self.last_total_tokens} | Price: ${self.last_price}\n"
            )
            log_file.write("=" * 60 + "\n")

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
        completion = self.completion(messages)
        answer_txt = completion.choices[0].message.content
        self.append_to_csv(
            question,
            prompt,
            "",
            answer_txt,
            completion.choices[0].finish_reason,
            completion.usage.completion_tokens,
            completion.usage.prompt_tokens,
            completion.usage.total_tokens,
            messages,
            completion,
        )
        self.to_file(messages, answer_txt)

        return answer_txt

    def append_to_csv(
        self,
        input_text,
        system_prompt_name,
        role,
        content,
        finish_reason,
        completion_tokens,
        prompt_tokens,
        total_tokens,
        messages_sent,
        completion,
        file_path_csv="./logs/current_conv.csv",
    ):
        """
        Appends conversation data to a CSV file.

        Args:
            input_text (str): The text input provided by the user.
            system_prompt_name (str): The name of the system prompt.
            role (str): The role of the message (i.e., user or system).
            content (str): The content of the message.
            finish_reason (str): The reason the conversation was finished.
            completion_tokens (int): The number of tokens generated by the completion.
            prompt_tokens (int): The number of tokens in the prompt.
            total_tokens (int): The total number of tokens used in the conversation.
            messages_sent (list): The messages sent during the conversation.
            completion (openai.Completion): The completion object returned by OpenAI.
            file_path_csv (str, optional): The file path for the CSV file. Defaults to "./logs/current_conv.csv".

        Returns:
            pandas.DataFrame: The DataFrame with the appended data.
        """
        row = {
            "class_id": id(self),
            "date": self.get_date_str(),
            "msg_type": input_text,
            "system_prompt_name": system_prompt_name,
            "role": role,
            "content": content,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "messages_sent": messages_sent,
            "completion": completion,
        }

        # Set up logging
        self.logger.debug(f"file_path_csv: {file_path_csv}")

        # Read existing CSV file or create a new one
        try:
            df = pd.read_csv(file_path_csv)
        except FileNotFoundError:
            df = pd.DataFrame()

        # Add the new row to the DataFrame and save to the CSV file
        new_df = pd.DataFrame(row, index=[0])
        df = pd.concat([df, new_df], ignore_index=True)
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

        # self.append_to_csv(completion, "prompt", self.last_total_tokens)
        return completion


# mc = MyChat()
