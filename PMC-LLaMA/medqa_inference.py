'''
Usage:

python medqa_inference.py \
    --model-name-or-path path/to/pmc_llama_model \
    --write-dir /path/to/inferenced_result_dir \
    --dataset-name path/to/dataset \
    --num-samples {number of samples [default: all data]} \
    --extract-entities \
    --augmentations synonyms paraphrase shuffle expand \
    --models-dir "/path/to/models" \ # Only use if models is not in the default directory
    --data-dir "/path/to/data" \ # Only use if data is not in the default directory
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
from augmentations import TextAugmentor

# clear cache if getting CUDA out of memory error
torch.cuda.empty_cache()

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
    parser.add_argument('--model-name-or-path', type=str, default="axiong/PMC_LLaMA_13B")
    parser.add_argument('--write-dir', type=str, default="inferenced_result_dir")
    parser.add_argument('--dataset-name', type=str, default="test_4_options.jsonl")
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--extract-entities', action='store_true', 
                       help='Enable entity extraction for questions and answers')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory containing model files for specific cache directory')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing data files for specific cache directory')
    parser.add_argument('--precision', type=str, default='float16',
                       help='Precision for model inference')
    parser.add_argument('--augmentations', nargs='+', 
                       choices=['synonyms', 'paraphrase', 'shuffle', 'expand', 'compress'],
                       help='Text augmentations to apply')
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
    """Resize tokenizer and embedding."""
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

def prepare_data(dataset: Sequence[dict], model, tokenizer, args) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prepared_data = []
    
    augmentor = TextAugmentor(model_name="llama3.2") if args.augmentations else None
    
    for idx, data_entry in enumerate(tqdm(dataset)):
        sample_id = f'sample{idx}'
        
        if 'instruction' in data_entry:
            prompt = prompt_no_input.format_map(data_entry)
            prepared_entry = {
                'sample_id': sample_id,
                'pmc_input': prompt,
                'question': '',
                'options': '',
                'augmentation_type': 'none'
            }
            prepared_data.append(prepared_entry)
        else:
            question = data_entry.get('question', '')
            options = data_entry.get('options', '')
            
            if not question or not options:
                print(f"Warning: Missing required fields for {sample_id}")
                continue
                
            
            try:
                # get original model response
                original_prompt = prompt_input.format_map({
                    'question': question,
                    'options': options
                })
                original_response = model.generate(
                    **tokenizer(original_prompt, return_tensors='pt').to(model.device)
                ).text
                
                # Generate augmentations if requested
                if augmentor and args.augmentations:
                    augmented_texts = augmentor.batch_augment([question], args.augmentations)
                    for aug_type, aug_text in zip(args.augmentations, augmented_texts):
                        aug_prompt = prompt_input.format_map({
                            'question': aug_text,
                            'options': options
                        })
                        
                        aug_entry = {
                            'sample_id': f'{sample_id}_{aug_type}',
                            'pmc_input': aug_prompt,
                            'question': question,  # Keep original question
                            'options': options,
                            'augmentation_type': aug_type,
                            'augmented_question': aug_text,  # Store augmented version here
                            'original_response': original_response
                        }
                        prepared_data.append(aug_entry)
                            
            except Exception as e:
                print(f"Error preparing entry {sample_id}: {str(e)}")
                continue
    
    return prepared_data

def save_results(write_dir: str, sample_id: str, model_output: str, 
                entities: list = None, original_data: dict = None) -> None:
    """Save inference results and question entities."""
    response_text = model_output.split("Response:")[-1].strip()
    
    results = {
        "original_data": {
            "question": original_data.get('question', ''),  # Original question
            "augmented_question": original_data.get('augmented_question', None),  # Augmented version if it exists
            "options": original_data.get('options', ''),
            "augmentation_type": original_data.get('augmentation_type', 'none'),
            "original_response": original_data.get('original_response', None)
        },
        "model_response": response_text,
    }
    
    if entities:
        results["question_entities"] = entities
        
    output_path = os.path.join(write_dir, f"{sample_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    print('Starting script')

    assert torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    tqdm.write(f'Using {device_count} GPU(s):')
    for i in range(device_count):
        tqdm.write(f' - {torch.cuda.get_device_name(i)}')

    args = parse_args()
    MODEL_DIR = args.models_dir
    DATADIR = args.data_dir
    if args.precision == 'float16':
        PRECISION = torch.float16
    elif args.precision == 'float32':
        PRECISION = torch.float32
    else:
        raise ValueError(f"Invalid precision: {args.precision}, must be 'float16' or 'float32'")

    print(f"\033[32mLoading Dataset\033[0m")
    if args.num_samples is None:
        dataset = load_dataset(path='json', 
                               data_files=args.dataset_name, 
                               split="train",
                               cache_dir=DATADIR)
    else:
        dataset = load_dataset(path='json', 
                               data_files=args.dataset_name, 
                               split=f"train[:{args.num_samples}]",
                               cache_dir=DATADIR)
    print(f"Loaded {len(dataset)} samples")

    extractor = None
    if args.extract_entities:
        print(f"\033[32mInitializing Entity Extractor\033[0m")
        extractor = EntityExtractor()

    print(f"\033[32mPrepare Data\033[0m")
    prepared_data = prepare_data(dataset, None, None, args)

    if not prepared_data:
        print("No valid data entries found. Please check your dataset.")
        exit(1)

    print(f"\033[32mLoad Checkpoint\033[0m")
    model = transformers.LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        force_download=False,
        cache_dir=MODEL_DIR,
        device_map='auto',
        torch_dtype=PRECISION
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        force_download=False,
        cache_dir=MODEL_DIR,
        model_max_length=1000,
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
    for data_entry in tqdm(prepared_data):
        sample_id = data_entry['sample_id']
        input_str = [data_entry['pmc_input']]
        
        try:
            output_str = inference_on_one(input_str, model, tokenizer, generation_config)
            
            if extractor:
                try:
                    # Extract entities from original question
                    question = data_entry.get('question', '')
                    entities = extractor.extract_entities(question)
                    
                    original_data = {
                        'question': data_entry.get('question', ''),
                        'augmented_question': data_entry.get('augmented_question', None),
                        'options': data_entry.get('options', ''),
                        'augmentation_type': data_entry.get('augmentation_type', 'none')
                    }
                    save_results(args.write_dir, sample_id, output_str, entities, original_data)
                except Exception as entity_error:
                    print(f"Entity extraction error for sample {sample_id}: {entity_error}")
                    save_results(args.write_dir, sample_id, output_str)
            else:
                original_data = {
                    'question': data_entry.get('question', ''),
                    'augmented_question': data_entry.get('augmented_question', None),
                    'options': data_entry.get('options', ''),
                    'augmentation_type': data_entry.get('augmentation_type', 'none')
                }
                save_results(args.write_dir, sample_id, output_str, None, original_data)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Question text: {data_entry.get('question', '')}")

    print("Inference completed.")