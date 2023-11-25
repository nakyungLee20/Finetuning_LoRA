# Fine-Tune Llama2-13b on SE paired dataset -> llama2-13b-chat-hf
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_config, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="train_dataset", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    subset: Optional[str] = field(default=None, metadata={"help": "the subset to use"})
    # size_valid_set: Optional[int] = field(default=300, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=500, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=3980, metadata={"help": "the sequence length"})

    max_steps: Optional[int] = field(default=4000, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=20, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})

    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=4, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=500, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_8bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./output_1125", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})



def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        # print(example)
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = example['text']
    return text


def create_datasets(tokenizer, args):
    data = load_dataset(path=args.dataset_name,
                        data_files={"train": 'train_data_nogold.json', "dev": 'dev_data_nogold.json', "test": 'test_data_nogold.json'},
                        )

    train_data = data['train']
    valid_data = data["dev"]

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


# 0. Define trainer arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# 1. Define bnb config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. Define Lora config
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Load last checkpoint adaptor model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    # device_map={"": 0},
    device_map='auto',
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.load_adapter("/home/guest/lnk/llama/PEFT/output_1122/final_checkpoint")
base_model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained("/home/guest/lnk/llama/PEFT/output_1122/")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# 4. Load Dataset
train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

# 5. Define trainer with arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    eval_steps=500,
    remove_unused_columns=False,
    save_total_limit=5,
    run_name="llama2_chat_1125",
)

trainer = SFTTrainer(
    model= base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=True,
    max_seq_length=3980,
    tokenizer=tokenizer,
    args=training_args,
)

# For memory, empty cache
torch.cuda.empty_cache()

# 6. Train model and save final model
trainer.train(resume_from_checkpoint="/home/guest/lnk/llama/PEFT/output_1122/final_checkpoint")
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

