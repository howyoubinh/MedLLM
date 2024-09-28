import os
import transformers
import torch
from datasets import load_dataset
from time import time
import warnings
from tqdm import tqdm

# Turn off the future warnings because they're annoying
warnings.simplefilter(action='ignore', category=FutureWarning)

## User defined variables
REPO_PATH = '/home/binh/Projects/Medical_LLM/PMC-LLaMA' # Path to the repository
MODELS = ['axiong/PMC_LLaMA_13B',
          'chaoyi-wu/PMC_LLAMA_7B_10_epoch', 
          'chaoyi-wu/PMC_LLaMA_7B']
MODEL_NAME = MODELS[2] # Choose the model to load
DATASETS = ['medmcqa']
DATASET_NAME = DATASETS[0]
PRECISION = torch.float16
MODEL_DIR = os.path.join(REPO_PATH,'models') # Where huggingface saves the model
os.environ['HF_HOME'] = MODEL_DIR # Not sure if this is required, but sets the hugging face environment variable
DATA_DIR = os.path.join(REPO_PATH,'data') # Where the datasets are downloaded
RESULTS_DIR = os.path.join(REPO_PATH,'results') # Where to save the results

## Define the test input string
prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)
test_example = {
    "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    "input": (
        "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
        "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
        "She otherwise feels well and is followed by a doctor for her pregnancy. "
        "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
        "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
        "Which of the following is the best treatment for this patient?"
        "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
    )
}
test_string = [prompt_input.format_map(test_example)]

def print_memory_usage(stage):
    maxs = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert from bytes to GB
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # Convert from bytes to GB
        tqdm.write(f"[{stage}] GPU {i}: Memory allocated: {allocated:.2f} GB, Max memory allocated: {max_allocated:.2f} GB")
    tqdm.write(f"[{stage}] Total memory allocated: {sum(maxs) / (1024 ** 3):.2f} GB, Total max memory allocated: {sum(maxs) / (1024 ** 3):.2f} GB")

def load_tokenzier(model_name, model_dir, verbose = True):
    if verbose: 
        tqdm.write('Loading Tokenizer')
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path = model_name, 
        force_download=False,
        cache_dir=model_dir
    )
    if verbose: 
        print_memory_usage('After loading tokenizer')
    
    return tokenizer

def load_model(model_name, model_dir, precision, verbose = True):
    if verbose:
        tqdm.write('Loading Model')
    model = transformers.LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_name, 
    force_download=False, 
    cache_dir=model_dir, 
    device_map="auto",  # Automatically splits the model across multiple GPUs
    torch_dtype=precision  # Use FP16 precision to reduce memory usage
    )
    if verbose:
        print_memory_usage('After loading model')
    return model

def run_test_prompt(model, tokenizer, input_str, verbose = True):
    if verbose:
        tqdm.write('Running Test Prompt')

    # Tokenize the input string
    if verbose:
        tqdm.write('Tokenizing input string')
    model_inputs = tokenizer(
        input_str,
        return_tensors='pt',
        add_special_tokens=False
    )

    # Create a GenerationConfig object
    generation_config = transformers.GenerationConfig(
        max_new_tokens=1000,
        do_sample=True,
        top_k=50
    )

    # Generate the output
    if verbose:
        tqdm.write('Generating output')
    start_time = time()
    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        generation_config=generation_config
    )
    output_str = tokenizer.batch_decode(topk_output)
    tqdm.write(f'Model prediction: {output_str[0]}')
    # tqdm.write('model predict: ', output_str[0])
    tqdm.write(f'Elapsed time: {time() - start_time:.2f} seconds')

def load_dataset(dataset_name, cache_dir):
    dataset = load_dataset(
        dataset_name,
        cache_dir=cache_dir
    )
    return dataset

def run_on_dataset(model, tokenizer, dataset_name, verbose = True):
    # Load the dataset
    if verbose:
        tqdm.write(f'Loading dataset {dataset_name}')
    dataset = load_dataset(dataset_name, DATA_DIR)

    #TODO: Add code to run the model on the dataset

# Confirm we're using CUDA and print GPU information
assert torch.cuda.is_available()
device_count = torch.cuda.device_count()
tqdm.write(f'Using {device_count} GPU(s):')
for i in range(device_count):
    tqdm.write(f' - {torch.cuda.get_device_name(i)}')

try:
    tokenizer = load_tokenzier(MODEL_NAME, MODEL_DIR, verbose=True)
except Exception as e:
    tqdm.write(f'Error loading tokenizer for model {MODEL_NAME}: {e}')
    raise e

try:
    model = load_model(MODEL_NAME, MODEL_DIR, PRECISION, verbose=True)
except Exception as e:
    tqdm.write(f'Error loading model {MODEL_NAME}: {e}')
    raise e
    
try:
    run_test_prompt(model, tokenizer, test_string)
except Exception as e:
    tqdm.write(f'Error running test prompt for model {MODEL_NAME}: {e}')

tqdm.write(f"Test prompt for model {MODEL_NAME} completed")    
print_memory_usage('End of test prompt')