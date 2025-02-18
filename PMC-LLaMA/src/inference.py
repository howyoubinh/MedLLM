import transformers
from typing import Dict, Optional, Sequence, List
from tqdm import tqdm

## Formatting for the PMC LLaMA inputs

def construct_special_tokens_dict(tokenizer) -> dict:
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
    pad_to_multiple_of: Optional[int] = None,
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

def inference_on_one(input_str: Sequence[str], model, tokenizer,generation_config) -> str:
    device = next(model.parameters()).device
    model_inputs = tokenizer(   input_str,
                                return_tensors='pt',
                                padding=True,
                            ).to(device)
    
    
    topk_output = model.generate(   model_inputs.input_ids,
                                    attention_mask=model_inputs.attention_mask,
                                    generation_config=generation_config,
                                    )
    output_str = tokenizer.batch_decode(topk_output)[0]  # a list containing just one str
    return output_str

def inference_on_many(input_strs: Sequence[str], model, tokenizer, generation_config, batch_size: int = 8) -> List[str]:
    """
    Perform inference on multiple input strings in a batched format.

    Args:
        input_strs (Sequence[str]): List of input strings to process.
        model: The model used for inference.
        tokenizer: The tokenizer for preprocessing input strings.
        generation_config: Configuration for generation (e.g., max length, top-k).
        batch_size (int): Number of inputs to process per batch.

    Returns:
        List[str]: List of generated output strings.
    """
    device = next(model.parameters()).device
    results = []

    # Process inputs in batches
    for i in tqdm(range(0, len(input_strs), batch_size)):
        batch_inputs = input_strs[i:i + batch_size]

        # Tokenize the batch
        model_inputs = tokenizer(
            batch_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(device)

        # Generate outputs for the batch
        topk_outputs = model.generate(
                                    model_inputs.input_ids,
                                    attention_mask=model_inputs.attention_mask,
                                    generation_config=generation_config)

        # Decode and append results
        batch_outputs = tokenizer.batch_decode(topk_outputs)
        results.extend(batch_outputs)

    return results
    