'''
Usage:

python qa_inference_hf.py \
    --model-name-or-path path/to/pmc_llama_model \
    --write-dir /path/to/inferenced_result_dir \
    --dataset-name path/to/dataset \
    --num-samples {number of samples [default: all data]}
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

# clear cache if getting CUDA out of memory error
torch.cuda.empty_cache()

MODELDIR  = 'models'
DATADIR = 'data'
PRECISION = torch.float16

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default="axiong/PMC_LLaMA_13B") # change this to your actual directory
    parser.add_argument('--write-dir', type=str, default="inferenced_result_dir")
    parser.add_argument('--dataset-name', type=str, default="axiong/pmc_llama_instructions") # download this from https://huggingface.co/datasets/axiong/pmc_llama_instructions
    parser.add_argument('--num-samples', type=int) # number of samples
    args = parser.parse_args()
    return args


def construct_spedical_tokens_dict() -> dict:
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
        sample_num = idx // 3 + 1
        sample_letter = chr(97 + idx % 3)  # 'a', 'b', or 'c'
        sample_id = f'sample{sample_num}{sample_letter}'
        
        if 'instruction' not in data_entry or 'input' not in data_entry:
            print(f"Warning: Missing 'instruction' or 'input' for {sample_id}. Skipping.")
            continue
        
        prompt = prompt_input.format_map(data_entry) if data_entry.get("input", "") != "" else prompt_no_input.format_map(data_entry)
        
        prepared_entry = {
            'sample_id': sample_id,
            'pmc_input': prompt
        }
        prepared_data.append(prepared_entry)
    
    return prepared_data



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
        dataset = load_dataset(args.dataset_name, split="train",cache_dir = DATADIR)
    else:
        dataset = load_dataset(args.dataset_name, split=f"train[:{args.num_samples}]",cache_dir = DATADIR)
    print(f"Loaded {len(dataset)} samples")

    print(f"\033[32mPrepare Data\033[0m")
    fn = partial(prepare_data, model=None, tokenizer=None)
    inference_data = fn(dataset)

    if not inference_data:
        print("No valid data entries found. Please check your dataset.")
        exit(1)

    print(f"\033[32mLoad Checkpoint\033[0m")
    model = transformers.LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path = args.model_name_or_path, 
    force_download=False, 
    cache_dir=MODELDIR, 
    device_map="auto",  # Automatically splits the model across multiple GPUs
    torch_dtype=PRECISION  # Use FP16 precision to reduce memory usage
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path = args.model_name_or_path, 
        force_download=False, 
        cache_dir=MODELDIR,
        model_max_length=1000,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = construct_spedical_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # model.cuda()
    # generation_config = create_generation_config()
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
            with open(os.path.join(args.write_dir, f"{sample_id}.txt"), 'w') as f:
                f.write(output_str)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")

    print("Inference completed.")