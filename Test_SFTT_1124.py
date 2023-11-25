import torch
import torch.nn.functional as F
import json
from peft import PeftModel

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from typing import List, Literal, Optional, Tuple, TypedDict

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


## Load fine-tuned model ##
model_name = "meta-llama/Llama-2-13b-chat-hf"
adapters_name = "output_1122/final_checkpoint"

ft_model = LlamaForCausalLM.from_pretrained(
    model_name,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    # device_map={"": 0},
    device_map='auto'
)

ft_model = PeftModel.from_pretrained(ft_model, adapters_name)
ft_model = ft_model.merge_and_unload()

ft_tokenizer = AutoTokenizer.from_pretrained("output_1122")
ft_tokenizer.pad_token = ft_tokenizer.eos_token

print("Finish loading model and tokenizer.")


## Load data and prompts ##
with open("train_dataset/test_data_nogold.json", 'r') as file:
    input_data = json.load(file)

# few-shot example
with open("train_dataset/train_data_nogold.json", 'r') as file:
    example = json.load(file)

B_INST, E_INST = "[INST]", "[/INST]"
example_prompt = example[1692]["text"]
example_prompt += "\n"
# print(example_prompt)

queries =[]
for input in input_data:
    arr = input['text'].split("<</SYS>>\n\n")
    query = arr[1].split("[/INST]")[0]
    concat = example_prompt + f"{B_INST} {query.strip()} {E_INST}"
    # print(concat)
    queries.append(concat)

print("Finish loading test data.")


# inference code
output =[]
for test in tqdm(queries):
    tokens = ft_tokenizer(test, return_tensors="pt", padding =True).to("cuda")

    generated_tokens = ft_model.generate(**tokens,
                                         temperature=0.3,
                                         top_p=0.9,
                                         max_new_tokens=4096)

    out=ft_tokenizer.decode(generated_tokens[0])
    # print(out)
    output.append(out)

print("Finish generating.")


# store result
with open("ft_result_1124_nogold.json","w",encoding='utf-8') as file:
    json.dump(output,file)

