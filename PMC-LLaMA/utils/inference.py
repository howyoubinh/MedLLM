import jsonlines
import argparse
import torch.backends
import transformers
import torch
import os
from typing import Dict, Optional, Sequence
from tqdm import tqdm
import time



instruction = "You're a doctor, kindly address the medical queries according to the patient's account. Analyze the question by option and answer with the best option."

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal Performance Shaders for macOS
    else:
        return torch.device("cpu")  # Fallback to CPU

## Formatting for the PMC LLaMA inputs
def read_jsonl(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list

def format_input_string(instruction:str,dataset_item:dict):
    preamble = "Below is an instruction that describes a task, paired with an input that provides further context. "
    formatted_input = (f"{preamble}\n\n"
                       f"### Instruction:\n{instruction}\n\n"
                       f"### Question:\n{dataset_item['Question']}\n"
                       f"### Options:\n{dataset_item['Options']}\n### Response:\n")
    return formatted_input

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
    pad_to_multiple_of: Optional[int] = 64,
):
    """
    Resize tokenizer and embedding with optional padding for Tensor Core optimization.
    Args:
        special_tokens_dict (Dict): Dictionary of special tokens to add.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to resize.
        model (transformers.PreTrainedModel): The model whose embeddings are resized.
        pad_to_multiple_of (Optional[int]): If specified, ensures embedding size is a multiple of this value.
    Note:
        Padding the embedding size to a multiple of 64 can improve performance on NVIDIA Tensor Cores.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_vocab_size = len(tokenizer)
    if pad_to_multiple_of:
        # Calculate padding to make vocab size a multiple of pad_to_multiple_of
        padding = (pad_to_multiple_of - new_vocab_size % pad_to_multiple_of) % pad_to_multiple_of
        new_vocab_size += padding
    model.resize_token_embeddings(new_vocab_size)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data if model.get_output_embeddings() else None
        # Average embeddings for new tokens
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        if output_embeddings is not None:
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return model

def prepare_data(data_list: Sequence[dict]) -> Sequence[dict]:
    for _idx in tqdm(range(len(data_list))):
        formatted_string = format_input_string(instruction,data_list[_idx])
        data_list[_idx]['pmc_input'] = formatted_string
    return data_list

def inference_on_one(input_str: Sequence[str], model, tokenizer,generation_config) -> str:
    device = next(model.parameters()).device
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    ).to(device)
    topk_output = model.generate(
        model_inputs.input_ids,
        generation_config=generation_config
    )
    output_str = tokenizer.batch_decode(topk_output)[0]  # a list containing just one str
    return output_str

def inference_on_many(
    batch_prompts: Sequence[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig
) -> Sequence[str]:
    """Run a batch of prompts through the model and decode the outputs."""

    # Tokenize the batch prompts all at once
    inputs = tokenizer(
        batch_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    
    # Move everything to the device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )

    # Decode
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return decoded

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default="axiong/PMC_LLaMA_13B") # change this to your actual directory
    parser.add_argument('--write-dir', type=str, default="./PMC-LLaMA/results")
    parser.add_argument('--data-path', type=str, default="./PMC-LLaMA/data/formatted_datasets/medqa_test.jsonl") 
    parser.add_argument('--num-samples', type=int) # number of samples
    parser.add_argument('--model_dir', type=str, default= './PMC-LLaMA/models')
    parser.add_argument('--precision', type=str, default="16")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.precision == "b16":
        precision = torch.bfloat16
    elif args.precision == "16":
        precision = torch.float16
    elif args.precision == "32":
        precision = torch.float32
    else:
        raise ValueError("Precision must be either 16 or 32")

    # Load the data
    formatted_dataset_path = "PMC-LLaMA/data/formatted_datasets/medmcqa_test.jsonl"
    data_list = read_jsonl(formatted_dataset_path)
    data_list = prepare_data(data_list)


    # Load model and tokenizer, offloading to gpu where available
    device = get_device()
    print(f"\033[32mUsing device: {device}\033[0m")

    # Load the model
    start_time = time.time()
    print(f"\033[32mLoading Model\033[0m")
    model = transformers.LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        force_download=False,
        cache_dir=args.model_dir,
        device_map="auto" if device.type == "cuda" else None,  # Enable device mapping for GPUs
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # FP16 for CUDA
        offload_folder=os.path.join(args.model_dir, "offload_folder") if ((device.type == "cpu") | (device.type == "mps")) else None,
    )
    model = model.to(device)
    print(f"Time to load model: {time.time()-start_time}")

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path = args.model_name_or_path, 
        force_download=False, 
        cache_dir=args.model_dir,
        model_max_length=1000,
        padding_side="right",
        use_fast=False,
    )

    # Resize tokenizer and embedding
    special_tokens_dict = construct_special_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of= 64,
    )

    generation_config = transformers.GenerationConfig(
                max_new_tokens=1000,
                top_k=50
        )

    # Run a single inference - don't save the output yet
    print(f"\033[32mRunning Inference\033[0m")
    input_str = data_list[0]['pmc_input']
    print(input_str)
    start_time = time.time()
    output_str = inference_on_one(input_str, model, tokenizer, generation_config)
    print(output_str)
    print(f"\nInference Time: {time.time() - start_time}")