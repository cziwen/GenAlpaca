import abc
import logging
from typing import List, Dict, Any

class LLMApiBase(abc.ABC):
    @abc.abstractmethod
    def chat_completion(self, messages: List[Any],
                       model: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> str:
        pass

class OpenAILLMApi(LLMApiBase):
    def __init__(self, api_key: str):
        import openai
        # 关闭OpenAI库的HTTP请求日志
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = openai.OpenAI(api_key=api_key)

    def chat_completion(self, messages: List[Any],
                       model: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""

class DeepSeekLLMApi(LLMApiBase):
    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com/v1"):
        import openai
        # 关闭OpenAI库的HTTP请求日志
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        # base_url 兼容 deepseek
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base.replace("/v1", ""))

    def chat_completion(self, messages: List[Any],
                       model: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "" 