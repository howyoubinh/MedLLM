import jsonlines

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
                       f"### Options:\n{dataset_item['Options']}\n### Response:")
    return formatted_input

formatted_dataset_path = "PMC-LLaMA/data/formatted_datasets/medmcqa_test.jsonl"
data_list = read_jsonl(formatted_dataset_path)
print(format_input_string("Instruction goes here",data_list[0]))