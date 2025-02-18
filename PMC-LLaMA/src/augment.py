from src.classes.text_augmentor import TextAugmentor
from src import utils
from typing import Sequence

def initialize_augmentor(model:str="llama3.2")->TextAugmentor:
    device = utils.get_device()
    if device == "cpu":
        raise RuntimeWarning("WARNING: Errors have been found if not using GPU. Confirm CUDA is loaded")
    augmentor = TextAugmentor(model_name=model)
    return augmentor

def format_input_string(preamble:str,instruction:str,question:str,options:str)->str:
    formatted_input = (f"{preamble}\n\n"
                       f"### Instruction:\n{instruction}\n\n"
                       f"### Question:\n{question}\n"
                       f"### Options:\n{options}\n### Response:\n")
    return formatted_input

def prepare_data(
    data_list: Sequence[dict],
    augmentor=None,
    augmentation=None,
    preamble="DEFAULT_PREAMBLE",
    instruction="DEFAULT_INSTRUCTION",
    parallel:bool=False,
    num_workers:int=4,
) -> Sequence[dict]:
    """
    Batch-prepares data by:
      1) Extracting all questions at once.
      2) If `augmentor` is provided, calling `augmentor.batch_augment` on ALL questions in one batch.
      3) Storing the results (prompts, final augmented text, etc.) into each item.
    """
    # 1) Extract all questions up front
    all_questions = [item['Question'] for item in data_list]
    # 2) If no augmentor is provided  skip augmentation entirely
    if augmentor is None:
        for i in range(len(data_list)):
            data_list[i]['Augmentation'] = augmentation
            data_list[i]['Augmentation Prompts'] = []
            data_list[i]['Augmented Questions'] = []
            data_list[i]['pmc_input'] = format_input_string(
                preamble,
                instruction,
                data_list[i]['Question'],
                data_list[i]['Options']
            )
        return data_list
    # 3) If augmentor is provided, perform a *single* batch augmentation of all questions
    result_list = augmentor.batch_augment(texts=all_questions, 
                                          methods=augmentation,
                                          parallel=parallel,
                                          max_workers=num_workers)
    # 4) Store the outputs back into data_list
    for i in range(len(data_list)):
        data_list[i]['Augmentation'] = augmentation
        # Safety check if result_list has the same length as data_list
        if i < len(result_list):
            data_list[i]['Augmentation Prompts'] = result_list[i]['prompts']
            data_list[i]['Intermediate Questions'] = result_list[i]['augmented_texts']
            data_list[i]['Augmented Question'] = result_list[i]['final_augmented_text']
            data_list[i]['pmc_input'] = format_input_string(
                preamble,
                instruction,
                result_list[i]['final_augmented_text'],
                data_list[i]['Options']
            )
        else:
            # If, for some reason, the results don't line up
            data_list[i]['Augmentation Prompts'] = "N/A"
            data_list[i]['Augmented Questions'] = "N/A"
            data_list[i]['Intermediate Questions'] = "N/A"
            data_list[i]['pmc_input'] = format_input_string(
                preamble,
                instruction,
                data_list[i]['Question'],
                data_list[i]['Options']
            )
    return data_list