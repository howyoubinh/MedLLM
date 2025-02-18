from typing import List
import json
from pathlib import Path
import ollama
from nltk.tokenize import sent_tokenize
import random
import nltk
nltk.download('punkt_tab',quiet=True)
import subprocess
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class TextAugmentor:
    def __init__(self, model_name: str = "llama3.2"):
        """Initialize augmentor with Ollama's Llama3.2"""
        self.model_name = model_name
        self.prompts = self._load_prompts()
        counter = 0
        while not self.check_startup():
            counter +=1
            print(f"Attempt number{counter}")
            if counter >=3:
                raise RuntimeError("Cannot start Ollama")

    def check_startup(self) -> bool:
        try:
            # Check if Ollama is already running
            print("Checking if OLLAMA is working")
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": "This is a message to check if you're working. If you are, write me a haiku about how much you love me"}])
            print(f"\n\n{response.message.content}\n\nOLLaMA startup successful")
            return True
        except Exception as e:
            print(f"{e}\n\nOllama is not running. Attempting to start it...")
            subprocess.Popen(
                ["./PMC-LLaMA/models/ollama/bin/ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(5)  # Give some time for the server to start
            return False
   
    def _load_prompts(self) -> dict:
        prompt_path = Path(__file__).parent / 'prompts.json'
        try:
            with open(prompt_path, 'r') as f:
                prompt_dict = json.load(f)
            return prompt_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"prompts.json not found at {prompt_path}")
             
    def _generate_response(self, prompt: str) -> dict:
        """Generate response using Ollama API with a system message for formatting."""
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", 
                 "content": "You are a question augmentation model. "
                            "Return the augmented question without explanations, preambles, or formatting. "
                            "If the original question is styled as a case history, ensure the augmented question is similarly styled. "
                            "The response must be a question. "
                            },
                {"role": "user", "content": prompt}
            ]
        )
        augmented_question = response['message']['content'].strip()
        augmented_dict = {'prompt': prompt,
                    'Augmented Question': augmented_question}
        return augmented_dict

    def synonyms(self, text: str) -> dict:
        prompt_template = self.prompts['synonyms']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def paraphrase(self, text: str) -> dict:
        prompt_template = self.prompts['paraphrase']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def shuffle(self, text: str) -> dict:
        # sentences = sent_tokenize(text)
        # if len(sentences) <= 1:
        #     return text
        # random.shuffle(sentences)
        # shuffled_text = ' '.join(sentences)
        prompt_template = self.prompts['shuffle']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=shuffled_text)}"
        return self._generate_response(prompt)

    def expand_context(self, text: str) -> dict:
        prompt_template = self.prompts['expand']
        prompt = f"{prompt_template['instruction']}\n\n{prompt_template['format'].format(text=text)}"
        return self._generate_response(prompt)

    def batch_augment(
        self,
        texts: List[str],
        methods: List[str],
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[dict]:

        def augment_single(text: str) -> dict:
            current_text = text
            prompts_list = []
            augmented_texts_list = []

            for method in methods:
                if hasattr(self, method):
                    result_dict = getattr(self, method)(current_text)
                    prompts_list.append(result_dict["prompt"])
                    augmented_texts_list.append(result_dict["Augmented Question"])
                    current_text = result_dict["Augmented Question"]

            if not prompts_list and not augmented_texts_list:
                prompts_list.append(text)
                augmented_texts_list.append(text)
                current_text = text

            return {
                "original_text": text,
                "prompts": prompts_list,
                "augmented_texts": augmented_texts_list,
                "final_augmented_text": current_text
            }

        if parallel:
            # ---- PARALLEL EXECUTION WITH A PROGRESS BAR ----
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create one future per text
                futures = [executor.submit(augment_single, text) for text in texts]

                # Use as_completed to update tqdm as each future finishes
                for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting in parallel"):
                    results.append(future.result())

            return results
        else:
            # ---- SEQUENTIAL EXECUTION WITH A PROGRESS BAR ----
            augmented_results = []
            for text in tqdm(texts, desc="Augmenting sequentially"):
                augmented_results.append(augment_single(text))
            return augmented_results