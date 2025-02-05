
import argparse
from src import utils, dataset_loader, augment, inference
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', type=str, default="PMC-LLaMA/configs/medqa-test_raw.yml") # change this to your actual directory
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config_dict = utils.load_config(args.config_filepath)

    system_config = config_dict['system']
    experiment_config = config_dict['experiment']
    dataset_config = config_dict['dataset']
    augmentation_config = config_dict['augmentation']
    inference_config = config_dict['inference']

    ############################################
    #####     Set the output filenames     #####
    ############################################
    if experiment_config['name'] == '':
        cache_path = os.path.join(system_config['cache_path'])
        results_path = os.path.join(system_config['results_path'])
        evaluation_path = os.path.join(system_config['evaluation_path'])
    else:
        cache_path = os.path.join(system_config['cache_path'],experiment_config['name'])
        results_path = os.path.join(system_config['results_path'], experiment_config['name'])
        evaluation_path = os.path.join(system_config['evaluation_path'],experiment_config['name'])

    if augmentation_config['do_augment']:
        preprocessed_filename = '_'.join([dataset_config['name'],dataset_config['set'],'-'.join(augmentation_config['augmentations'])])+'.jsonl'
    else:
        preprocessed_filename = '_'.join([dataset_config['name'],dataset_config['set'],'raw'])+'.jsonl'

    ########################################################################
    #####     Check the path to the raw dataset exists if required     #####
    ########################################################################
    if dataset_config['custom_dataset_path'] == '':
        raw_dataset_path = os.path.join(system_config['data_path'],
                                    f"formatted_datasets",
                                    f"{dataset_config['name']}_{dataset_config['set']}.jsonl")
        if not os.path.exists(raw_dataset_path) and augmentation_config['do_augment']:
            dataset_loader.format_datasets(dataset_list=[dataset_config['name']],
                                           cache_dir=os.path.abspath(system_config['data_path']),
                                           overwrite=False)
            if not os.path.exists(raw_dataset_path):
                raise FileNotFoundError(f"Something went wrong loading the dataset: {dataset_config['name']}")
    else:
        raw_dataset_path = dataset_config['custom_dataset_path']
        if not os.path.exists(raw_dataset_path) and augmentation_config['do_augment']:
            raise FileNotFoundError(f"Could not find .jsonl data at {raw_dataset_path}")
    
    ################################
    #####     Augmentation     #####
    ################################
    preprocessed_path = os.path.join(cache_path, preprocessed_filename)
    if (not os.path.exists(preprocessed_path)) or augmentation_config['overwrite']:

        os.makedirs(cache_path,exist_ok=True)
        if augmentation_config['do_augment']:
            try:
                augmentor = augment.initialize_augmentor(augmentation_config['model'])
            except Exception as e:
                print(f"Error loading augmentor\n{e}")
                raise e
        else:
            augmentor = None 

        raw_dataset = utils.load_raw_dataset(raw_dataset_path=raw_dataset_path)
        print(f"output_filepath = {preprocessed_path}")
        preprocessed_data = augment.prepare_data(data_list=raw_dataset[0:50],
                                                augmentor=augmentor,
                                                augmentation=augmentation_config['augmentations'],
                                                preamble=augmentation_config['preamble'],
                                                instruction=augmentation_config['instruction'],
                                                parallel=augmentation_config['parallel'],
                                                num_workers=augmentation_config['num_workers'])
        utils.write_jsonl(preprocessed_data,preprocessed_path)
        print(f"Dataset preprocessed and saved to {preprocessed_path}")

    #############################
    #####     Inference     #####
    #############################
    if inference_config['do_inference']:
        import torch
        import transformers
        torch.cuda.empty_cache()

        print(f"Loading data from cached path:{preprocessed_path}")
        data = utils.read_jsonl(preprocessed_path)
        print(f"Loaded {len(data)} samples")
        if (len(data) < inference_config['num_samples']) | (inference_config['num_samples']<0):
            data = data
        else:
            data = data[0:inference_config['num_samples']]
        print(f"Selected {len(data)} samples for inference")
        input_strings = list()
        for entry in data:
            input_strings.append(entry['pmc_input'])

        if inference_config['precision'] == "b16":
            precision = torch.bfloat16
        elif inference_config['precision'] == "16":
            precision = torch.float16
        elif inference_config['precision'] == "32":
            precision = torch.float32
        else:
            raise ValueError("Precision must be either 'b16', '16' or '32'")

        device = utils.get_device()

        print(f"\n\nLoading model from pretrained\n\n")
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=inference_config['model'],
            force_download=False,
            cache_dir=system_config['model_path'],
            device_map="auto" if device.type == "cuda" else None,  # Enable device mapping for GPUs
            torch_dtype=precision, 
            offload_folder=os.path.join(system_config['model_path'], "offload_folder") if ((device.type == "cpu") | (device.type == "mps")) else None,
            )
        
        if device.type != "cuda":
            model = model.to(device)

        print(f"\n\nSetting up tokenizer\n\n")
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path = inference_config['model'], 
            force_download=False, 
            cache_dir=system_config['model_path'],
            model_max_length=1000,
            padding_side="right",
            use_fast=False,
            )
        
        print(f"\n\nResizing embeddings\n\n")
        special_tokens_dict = inference.construct_special_tokens_dict(tokenizer=tokenizer)
        model = inference.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of= 64,
        )

        print(f"\n\nDefining Generation config\n\n")
        generation_config = transformers.GenerationConfig(
                max_new_tokens=1000,
                top_k=50
        )
     
        if inference_config['batch_size'] > 1:
            print(f"\n\nRunning Inference on Many\n\n")
            output_strings=inference.inference_on_many(input_strs=input_strings,
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    generation_config=generation_config,
                                                    batch_size=inference_config['batch_size'])
        else:
            print(f"\n\nRunning Inference on One\n\n")
            output_strings = []
            for input in tqdm(input_strings):
                output = inference.inference_on_one(input_str=input_strings,
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    generation_config=generation_config)
                output_strings.append(output)
        
        for i, entry in tqdm(enumerate(data)):
            entry['pmc_output'] = output_strings[i]

        os.makedirs(results_path,exist_ok=True)
        run_num = len([f for f in os.listdir(results_path) if 'run' in f]) + 1
        results_filepath = os.path.join(results_path,f"run_{run_num}_{os.path.splitext(preprocessed_filename)[0]}.jsonl")
        utils.write_jsonl(data,results_filepath)
        print(f"Results saved to {results_filepath}")

        
        
        

        
