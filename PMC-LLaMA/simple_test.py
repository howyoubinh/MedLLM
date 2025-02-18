import transformers
import torch
# tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
# model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')
# use 7B model
tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
sentence = 'Hello, doctor' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=200, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))
