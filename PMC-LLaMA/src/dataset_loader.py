"""
These helper functions are used to load the datasets from hugging face in a standard format.
Run format_datasets with a list of dataset names to cache the formatted datasets.
The current supported datasets are:
- "medqa": GBaker/MedQA-USMLE-4-options
- "medmcqa": openlifescienceai/medmcqa
Currently in the works:
- "pubmedqa": qiaojin/PubMedQA

The format_datasets function will cache the formatted datasets in the specified directory.
The default directory is "./PMC-LLaMA/data/formatted_datasets".

"""
from typing import Tuple
from datasets import load_dataset, Dataset
from tqdm import tqdm
import jsonlines
import os

## Loaders
def get_dataset(dataset_name,config_name=None,cache_dir:str="./PMC-LLaMA/data")->Dataset:
    """
    Load a dataset from the Hugging Face Datasets library.
    Currently supported datasets:
    - "medqa": GBaker/MedQA-USMLE-4-options
        default_config: None
    - "medmcqa": openlifescienceai/medmcqa
        default_config: None
    - "pubmedqa": qiaojin/PubMedQA
        default_config: "pqa_labeled"
    Args:
    - dataset_name (str): The name of the dataset to load.
    - cache_dir (str): The directory to cache the dataset in.

    Returns:
    - dataset (datasets.Dataset): The loaded dataset.
    """
    if dataset_name == "medqa":
        dataset_url = "GBaker/MedQA-USMLE-4-options"
        config_name = None
    elif dataset_name == "medmcqa":
        dataset_url = "openlifescienceai/medmcqa"
        config_name = None
    elif dataset_name == "pubmedqa":
        if config_name == None:
            print("Loading the pqa_labeled config for PubMedQA. \nOther available configs are: 'pqa_artificial', 'pqa_unlabeled'")
            config_name = "pqa_labeled"
        dataset_url="qiaojin/PubMedQA"
    else:
        raise ValueError(f"{dataset_name} not yet implemented")
    dataset = load_dataset(dataset_url,config_name,cache_dir=cache_dir)
    return dataset # type: ignore

def medqa_format(item)->Tuple[str,str,str]:
    # Start with the question part
    question = item["question"] + "\n"
    answer = item["answer_idx"]
    # Initialize an empty string for the options
    options = ""
    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())
        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]
            # Append the key and value to the options string with a newline for all except the last one
            options += f"{key}. {value}\n"
    return question, options, answer

def medmcqa_format(item)->Tuple[str,str,str]:
    """
    The format the input of MedMCQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"
    # Initialize an empty string for the options
    options = "A. " + item["opa"] + "\n"
    options += "B. " + item["opb"] + "\n"
    options += "C. " + item["opc"] + "\n"
    options += "D. " + item["opd"] + "\n"
    answers = ['A','B','C','D']
    answer = answers[item['cop']]
    return question, options, answer

## Caching Function
def format_datasets(dataset_list:list,cache_dir:str="./PMC-LLaMA/data",overwrite:bool=False)->None:
    """
    Format datasets from the Hugging Face Datasets library into a common format.
    Args:
    - dataset_list (list): A list of dataset names to format.
    - cache_dir (str): The directory to cache the datasets in.
    - overwrite (bool): Whether to overwrite existing formatted datasets.
    Returns:
    - None
    """
    output_dir = os.path.join(cache_dir,"formatted_datasets")
    pbar = tqdm(dataset_list)
    for dataset_name in pbar:
        dataset = get_dataset(dataset_name)
        for key in dataset.keys():
            os.makedirs(output_dir,exist_ok=True)
            output_filepath = os.path.join(output_dir,f"{dataset_name}_{key}.jsonl")
            if os.path.exists(output_filepath) and not overwrite:
                print(f"Dataset {dataset_name}_{key} already exists. Skipping.")
                continue
            pbar.set_description(f"Formatting {dataset_name}_{key}")
            with jsonlines.open(output_filepath, mode='w') as writer:
                if dataset_name == "medqa":
                    formatter = medqa_format
                elif dataset_name == "medmcqa":
                    formatter = medmcqa_format
                else:
                    raise ValueError(f"{dataset_name} not yet implemented")
                for item in tqdm(dataset[key],leave=False):
                    question, options, answer = formatter(item)
                    row_dict = {"Dataset": dataset_name,
                                "Set": key,
                                "Question": question, 
                                "Options": options, 
                                "Answer": answer}
                    writer.write(row_dict)
    pbar.close()
    print(f"Formatted datasets saved to {output_dir}")
    return