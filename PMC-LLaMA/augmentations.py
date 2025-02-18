from typing import List, Dict
import json
from pathlib import Path
import ollama
from nltk.tokenize import sent_tokenize
import random
import nltk
nltk.download('punkt_tab')

class TextAugmentor:
    def __init__(self, model_name: str = "llama3.3"):
        """Initialize augmentor with one of Ollama's models"""
        self.model_name = model_name
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict:
        prompt_path = Path(__file__).parent / 'prompts.json'
        try:
            with open(prompt_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"prompts.json not found at {prompt_path}")
        
    def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        return response['message']['content'].strip()

    def synonyms(self, text: str) -> str:
        prompt_template = self.prompts['synonyms']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def paraphrase(self, text: str) -> str:
        prompt_template = self.prompts['paraphrase']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def shuffle(self, text: str) -> str:
        # sentences = sent_tokenize(text)
        # if len(sentences) <= 1:
        #     return text
        # random.shuffle(sentences)
        # shuffled_text = ' '.join(sentences)
        prompt_template = self.prompts['shuffle']
        # prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=shuffled_text)}"
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def expand(self, text: str) -> str:
        prompt_template = self.prompts['expand']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def batch_augment(self, texts: List[str], methods: List[str]) -> List[str]:
        augmented_texts = []
        for text in texts:
            augmented = text
            for method in methods:
                if hasattr(self, method):
                    augmented = getattr(self, method)(augmented)
            augmented_texts.append(augmented)
        return augmented_texts