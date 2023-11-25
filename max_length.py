import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments


model_name = "/home/broodling/llama/llama-2-13b-chat-hf"

data = load_dataset(path="/home/guest/lnk/llama/PEFT/train_dataset",
                    data_files={"train": 'train_data_nogold.json', "dev": 'dev_data_nogold.json', "test": 'test_data_nogold.json'})

train_data = data['train']
test_data = data['test']
dev_data = data["dev"]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# out=0
# outlier = []
tr_max = 0
tr_min = 3000
tr_min_idx = -1
for i, idx in enumerate(train_data):
    tokenized_input = tokenizer(idx['text'], truncation=False)["input_ids"]
    if(tr_max<len(tokenized_input)):
        tr_max = len(tokenized_input)

    if (tr_min > len(tokenized_input)):
        tr_min = len(tokenized_input)
        tr_min_idx = i

    #if(len(tokenized_input)>=4096):
    #    out += 1
    #    outlier.append(i)

print(tr_max)
print(tr_min)
print(tr_min_idx)
#print(out)
#print(outlier)


dev_min = 3000
min_idx = -1
dev_max = 0
for i, idx in enumerate(dev_data):
    tokenized_input = tokenizer(idx['text'], truncation=False)["input_ids"]
    if(dev_max<len(tokenized_input)):
        dev_max = len(tokenized_input)

    if(dev_min>len(tokenized_input)):
        dev_min = len(tokenized_input)
        min_idx = i

print(dev_max)
print(dev_min)
print(min_idx)


test_min = 3000
test_min_idx = -1
test_max =0
for i,idx in enumerate(test_data):
    tokenized_input = tokenizer(idx['text'], truncation=False)["input_ids"]
    if(test_max<len(tokenized_input)):
        test_max = len(tokenized_input)

    if (test_min > len(tokenized_input)):
        test_min = len(tokenized_input)
        test_min_idx = i

print(test_max)
print(test_min)
print(test_min_idx)


example_prompt = train_data[1692]["text"]
example_prompt += "\n"
print(example_prompt)

