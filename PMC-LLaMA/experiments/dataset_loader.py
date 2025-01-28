from datasets import load_dataset
from tqdm import tqdm
data_dir = "./PMC-LLaMA/data"


medmcqa_ds = load_dataset("openlifescienceai/medmcqa",cache_dir=data_dir)

for item in medmcqa_ds['train']:
    print(item)
    break

pubmedqa_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled",cache_dir=data_dir)

for item in pubmedqa_ds['train']:
    print(item)
    break

medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf",cache_dir=data_dir)

for item in medqa_ds['test']:
    print(item)
    break

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

for idx, data_entry in enumerate(tqdm(medmcqa_ds['train'])):
    sample_num = idx // 3 + 1
    print(data_entry)
    if sample_num > 3:
        break






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
