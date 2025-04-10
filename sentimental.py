from abc import ABC, abstractmethod
from typing import Dict, Union, List, Literal
import numpy as np


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers"""

    @abstractmethod
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of a single text. Returns a float from -1 to 1 where -1.0 is very negative, 0.0 is neutral, and 1.0 is very positive"""
        pass

    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of multiple texts. Returns a list of floats from -1 to 1 where -1.0 is very negative, 0.0 is neutral, and 1.0 is very positive"""
        return [self.analyze_sentiment(text) for text in texts]


class OpenAISentimentAnalyzer(SentimentAnalyzer):
    """Sentiment analyzer using OpenAI's API"""

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_sentiment(self, text: str) -> Dict[str, Union[float, str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'Analyze the sentiment of the following text. Return only a floating-point number where -1.0 is very negative, 0.0 is neutral, and 1.0 is very positive.'},
                {'role': 'user', 'content': text}
            ],
            temperature=0.0
        )

        return float(response.choices[0].message.content.strip())


def create_sentiment_analyzer(model_type: Literal['openai'], **kwargs) -> SentimentAnalyzer:
    if model_type == 'openai':
        if 'api_key' not in kwargs:
            raise ValueError(
                'OpenAI Sentiment Analyzer requires an api_key argument.')
        model = kwargs.get('model', 'gpt-4o-mini')  # < default OpenAI model
        return OpenAISentimentAnalyzer(api_key=kwargs['api_key'], model=model)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
