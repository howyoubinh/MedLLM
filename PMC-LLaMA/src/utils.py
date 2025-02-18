import jsonlines
from typing import List
import yaml
import os
import torch
from tqdm import tqdm

def load_config(yaml_file: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.
    
    :param yaml_file: Path to the YAML file.
    :return: Dictionary containing the YAML configuration.
    """
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file {yaml_file} was not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

def read_jsonl(file_path)->List:
    data_list = list()
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list

def load_raw_dataset(raw_dataset_path:str)->List:
    """Tries to load the dataset and checks the keys based on the first entry"""
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"{raw_dataset_path} does not exist")
    dataset_list = read_jsonl(raw_dataset_path)
    first_entry = dataset_list[0]
    required_keys = ['Dataset','Dataset','Set','Question','Options','Answer']
    for r_key in required_keys:
        if not r_key in first_entry.keys():
            raise KeyError(f"{r_key} missing from dataset list. Check dataset format")
    return dataset_list
        
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal Performance Shaders for macOS
    else:
        return torch.device("cpu")  # Fallback to CPU
    
def write_jsonl(data,output_filepath):
    with jsonlines.open(output_filepath, mode='w') as writer:
        for entry in tqdm(data):
            writer.write(entry)
    writer.close()
        