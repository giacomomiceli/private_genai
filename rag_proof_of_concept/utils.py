"""
Configuration tools module

This module provides functionality for handling configurations and settings stored in config files.

Components:
    - load_config: Function to load the configuration from a specified TOML file

Typical usage example:
    from config import load_config

    config = load_config('path/to/config.toml')
    print(config['some_setting'])

Note:
    This module requires the 'toml' library to be installed.
"""

# External imports
import logging
import toml
import random
from datetime import datetime
from enum import Enum
import requests
import json
import re
from openai import OpenAI
import tiktoken
from gradio_client import Client

from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

def load_config(config_file: str = 'config.toml') -> Dict[str, Any]:
    """
    Load the configuration from the specified file.

    Args:
        config_file (str, optional): The path to the configuration file. Defaults to 'config.toml'.
        
    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = toml.load(file)
    return config

def read_json(input: str) -> Any:
    # Remove everything to the left of the first '['
    input = re.sub(r'^.*?\[', '[', input)
    # Remove everything to the right of the last ']'
    input = re.sub(r'\].*$', ']', input)
    # Remove any remaining markdown code block delimiters
    input = re.sub(r"```\s*json\s*|```\s*", "", input)
    return json.loads(input)

class LLMClient:
    """
    Wrapper class for interacting with a language model which can communicate with the OpenAI API or a custom model served by a gradio or Flask app.
    
    Args:
        model (str): The model to use. Can be one of 'gpt-4o-mini', 'gpt-4o' (or other OpenAI models), or a URL to a gradio or Flask app.
    """
    def __init__(self, model):
        self._cloud_model_name = None
        if model in ['gpt-4o-mini', 'gpt-4o']:
            self.client = OpenAI()
            self._cloud_model_name = model
        elif re.match(r'^(https?://)', model):
            if 'gradio' in model:
                self.client = Client(model)
            else:
                # We assume it's served by flask
                self.client = model+'/api'
        else:
            raise ValueError(f"Model {model} not supported.")
        
        self._model = self._get_model_name_from_api()
        self._gpu = self._get_gpu_name_from_api()
        self._encoding = self._get_encoding()

    def _get_model_name_from_api(self) -> str:
        if isinstance(self.client, OpenAI):
            return self._cloud_model_name
        elif isinstance(self.client, Client):
            return self.client.predict(api_name="/model_name")
        else:
            # We assume it's served by flask
            response = requests.get(self.client + "/model_name")
            if response.status_code != 200:
                raise RuntimeError(f"The server at {self.client} did not return a valid response. Check the URL.")
            return response.text
        
    def _get_gpu_name_from_api(self) -> str:
        if isinstance(self.client, OpenAI):
            return 'unknown'
        elif isinstance(self.client, Client):
            return self.client.predict(api_name="/gpu_name")
        else:
            # We assume it's served by flask
            response = requests.get(self.client + "/gpu_name")
            if response.status_code != 200:
                raise RuntimeError(f"The server at {self.client} did not return a valid response. Check the URL.")
            return response.text
        
    def _get_encoding(self) -> Any:
        if isinstance(self.client, OpenAI):
            return tiktoken.encoding_for_model(self._model)
        else:
            return None
        
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def gpu(self) -> str:
        return self._gpu
    
    @property
    def max_tokens(self) -> int:
        return 128000 if self._model in ['gpt-4o', 'gpt-4o-mini'] else 128000
    
    def generate(self, prompt: Union[str, List[Dict[str, str]]], max_new_tokens: int = 512) -> str: 
        if isinstance(self.client, OpenAI):
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            return self.client.chat.completions.create(
                messages=prompt,
                model=self.model,
                max_tokens=max_new_tokens,
                temperature=0,
                n=1,
                stop=None).choices[0].message.content
        elif isinstance(self.client, Client):
            return self.client.predict(
                prompt=json.dumps(prompt) if isinstance(prompt, list) else prompt,
                max_new_tokens=max_new_tokens,
                api_name="/generate")
        else:
            # We assume it's served by flask
            response = requests.post(
                self.client + "/generate",
                json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            if response.status_code != 200:
                raise RuntimeError(f"The server at {self.client} did not return a valid response. Check the URL.")
            return response.text
    
    def _apply_chat_template(self, prompt: List[Dict[str, str]]) -> str:
        if isinstance(prompt, str):
            return prompt
        else :
            if isinstance(self.client, OpenAI):
                return "\n".join(f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>" for message in prompt)
            else:
                raise NotImplementedError("For models served by gradio or Flask, applying the chat template is performed on server side.")

    def _encode(self, prompt: Union[str, List[Dict[str, str]]]) -> Any:
        if isinstance(self.client, OpenAI):
            prompt = self._apply_chat_template(prompt)
            return self._encoding.encode(prompt)
        else:
            raise NotImplementedError("For models served by gradio or Flask, encoding is performed on server side.")
    
    def measure_prompt(
            self, 
            prompt: Union[str, List[Dict[str, str]]]) -> int:
        if isinstance(self.client, OpenAI):
            n_tokens = len(self._encode(prompt))
            return n_tokens, len(self._apply_chat_template(prompt))
        elif isinstance(self.client, Client):
            response = self.client.predict(
                prompt=json.dumps(prompt) if isinstance(prompt, list) else prompt, 
                api_name="/measure_prompt")
            return response['n_tokens'], response['n_chars']
        else: # We assume it's served by flask
            response = requests.post(
                self.client + "/measure_prompt",
                json={"prompt": prompt})
            if response.status_code != 200:
                raise RuntimeError(f"The server at {self.client} did not return a valid response. Check the URL.")
            return response.json()['n_tokens'], response.json()['n_chars']
        
    def estimate_prompt(
            self,
            prompt: Union[str, List[Dict[str, str]]]) -> int:
        fit_dict = {
            'gpt-4o-mini': [0, 1/4.5],
            'gpt-4o': [0, 1/4.5]
        }

        try:
            a, b = fit_dict[self.model]
        except KeyError:
            logger.warning(f"Model {self.model} not found in fit_dict. Using default values.")
            a, b = [0, 1/4.5]

        n_char = len(self._apply_chat_template(prompt))
        n_tokens = round(a + b * n_char)

        return n_tokens, n_char

    def token_count_to_str_length(self, n_tokens: int) -> float:
        fit_dict = {
            'gpt-4o-mini': [0, 4.5],
            'gpt-4o': [0, 4.5]
        }

        try:
            a, b = fit_dict[self.model]
        except KeyError:
            logger.warning(f"Model {self.model} not found in fit_dict. Using default values.")
            a, b = [0, 4.5]

        return a + b * n_tokens

    def validate_prompt_size(self, prompt: Union[str, List[Dict[str, str]]]) -> bool:
        n_tokens, _ = self.measure_prompt(prompt)
        return n_tokens <= self.max_tokens


# This class is used to restrict the possible values for the chunk granularity attribute.
class ChunkGranularity(Enum):
    RAW = "raw"
    CHUNK_SUMMARY = "chunk_summary"
    PAGE_SUMMARY = "page_summary"
    DOCUMENT_SUMMARY = "document_summary"


HEX_ID_LENGTH = 6

def generate_hex_ids(N: int):
    # Calculate the maximum value based on the length
    max_value = 16**HEX_ID_LENGTH - 1
    
    # Check if N is greater than the number of possible unique values
    if N > max_value + 1:
        raise ValueError(f"Cannot generate {N} unique hexadecimal numbers with length {HEX_ID_LENGTH}. Maximum possible is {max_value + 1}.")
    
    # Generate initial set of N random numbers
    initial_numbers = [random.randint(0, max_value) for _ in range(N)]
    unique_hex_numbers = {f"{num:0{HEX_ID_LENGTH}X}" for num in initial_numbers}
    
    # Continue generating numbers until we have N unique values
    while len(unique_hex_numbers) < N:
        random_number = random.randint(0, max_value)
        hex_number = f"{random_number:0{HEX_ID_LENGTH}X}"
        unique_hex_numbers.add(hex_number)
    
    return list(unique_hex_numbers)


def get_formatted_current_date() -> str:
    """
    Returns the current date formatted as "Weekday YYYY-MM-DD".
    """
    # Get the current date
    current_date = datetime.now()
    # Format the date
    formatted_date = current_date.strftime("%A %Y-%m-%d")
    return formatted_date