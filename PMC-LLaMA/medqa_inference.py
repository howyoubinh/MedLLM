'''
Usage:

python medqa_inference.py \
    --model-name-or-path path/to/pmc_llama_model \
    --write-dir /path/to/inferenced_result_dir \
    --dataset-name path/to/dataset \
    --num-samples {number of samples [default: all data]} \
    --extract-entities
'''

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers
from transformers import GenerationConfig

from typing import Dict, Optional, Sequence
import argparse
from tqdm import tqdm
from functools import partial
import os
from datasets import load_dataset
import json
from entity_extractor import EntityExtractor

# clear cache if getting CUDA out of memory error
torch.cuda.empty_cache()

# instruction = "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly."

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "Instruction:\nYou're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly after 'Response:'. Do not explain your answer.\n\n\
Question:\n{question}\n\n\
Options:\n{options}\n\n\
Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default="/home/binh/Projects/Medical_LLM/PMC-LLaMA/models/models--chaoyi-wu--PMC_LLaMA_7B/snapshots/6caf5c19bdcd157f9d9a7d374be66d7b61d75351") # change this to your actual directory (use 13B model)
    parser.add_argument('--write-dir', type=str, default="inferenced_result_dir")
    parser.add_argument('--dataset-name', type=str, default="test_4_options.jsonl")
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--extract-entities', action='store_true', 
                       help='Enable entity extraction for questions and answers') # add this argument to extract entities
    args = parser.parse_args()
    return args


def construct_special_tokens_dict() -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def inference_on_one(input_str: Sequence[str], model, tokenizer, generation_config) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        generation_config=generation_config
    )

    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]


def prepare_data(dataset: Sequence[dict], model, tokenizer) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prepared_data = []
    
    for idx, data_entry in enumerate(tqdm(dataset)):
        sample_id = f'sample{idx}'
        
        # Debug print
        # print(f"\nData entry {sample_id} contains keys: {data_entry.keys()}")
        # print(f"Data entry content: {data_entry}")
        
        if 'instruction' in data_entry:
            prompt = prompt_no_input.format_map(data_entry)
            prepared_entry = {
                'sample_id': sample_id,
                'pmc_input': prompt,
                'question': '',  # Empty string for no question
                'options': ''    # Empty string for no options
            }
        else:
            # Check if required fields exist
            question = data_entry.get('question', '')
            options = data_entry.get('options', '')
            
            if not question or not options:
                print(f"Warning: Missing or empty required fields for {sample_id}.")
                print(f"Question: {question}")
                print(f"Options: {options}")
                continue
                
            try:
                prompt = prompt_input.format_map({
                    'question': question,
                    'options': options
                })
                
                prepared_entry = {
                    'sample_id': sample_id,
                    'pmc_input': prompt,
                    'question': question,
                    'options': options
                }
            except Exception as e:
                print(f"Error preparing entry {sample_id}: {str(e)}")
                print(f"Data entry: {data_entry}")
                continue
        
        prepared_data.append(prepared_entry)
    
    return prepared_data

def save_results(write_dir: str, sample_id: str, model_output: str, 
                entities: list = None, original_data: dict = None) -> None:
    """Save inference results and question entities."""
    if entities:
        # Extract the actual response from model output (everything after "Response:")
        response_text = model_output.split("Response:")[-1].strip()
        
        results = {
            "original_data": {
                "question": original_data.get('question', ''),
                "options": original_data.get('options', '')
            },
            "model_response": response_text,
            "question_entities": entities  # Only storing question entities
        }
        output_path = os.path.join(write_dir, f"{sample_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Original txt file save behavior
        output_path = os.path.join(write_dir, f"{sample_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(model_output)



if __name__ == '__main__':
    print('Starting script')

    assert torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    tqdm.write(f'Using {device_count} GPU(s):')
    for i in range(device_count):
        tqdm.write(f' - {torch.cuda.get_device_name(i)}')

    args = parse_args()

    print(f"\033[32mLoading Dataset\033[0m")
    if args.num_samples is None:
        dataset = load_dataset('json', data_files=args.dataset_name, split="train")
    else:
        dataset = load_dataset('json', data_files=args.dataset_name, split=f"train[:{args.num_samples}]")
    print(f"Loaded {len(dataset)} samples")

    extractor = None
    if args.extract_entities:
        print(f"\033[32mInitializing Entity Extractor\033[0m")
        extractor = EntityExtractor()

    print(f"\033[32mPrepare Data\033[0m")
    fn = partial(prepare_data, model=None, tokenizer=None)
    inference_data = fn(dataset)

    if not inference_data:
        print("No valid data entries found. Please check your dataset.")
        exit(1)

    print(f"\033[32mLoad Checkpoint\033[0m")
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto'
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=400,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = construct_special_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    generation_config = GenerationConfig(
        max_new_tokens=1000,
        top_k=50
    )

    os.makedirs(args.write_dir, exist_ok=True)

    print(f"\033[32mRunning Inference\033[0m")
    for data_entry in tqdm(inference_data):
        sample_id = data_entry['sample_id']
        input_str = [data_entry['pmc_input']]
        
        try:
            output_str = inference_on_one(input_str, model, tokenizer, generation_config)
            
            if extractor:
                try:
                    # Only extract entities from the question
                    question = data_entry.get('question', '')
                    entities = extractor.extract_entities(question)
                    
                    original_data = {
                        'question': question,
                        'options': data_entry.get('options', {})
                    }
                    save_results(args.write_dir, sample_id, output_str, entities, original_data)
                except Exception as entity_error:
                    print(f"Entity extraction error for sample {sample_id}: {entity_error}")
                    print(f"Question text: {question}")
                    # Continue with basic save if entity extraction fails
                    save_results(args.write_dir, sample_id, output_str)
            else:
                save_results(args.write_dir, sample_id, output_str)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Question text: {data_entry.get('question', '')}")

    print("Inference completed.")