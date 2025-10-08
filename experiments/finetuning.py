"""
** Supervised fine-tuning code ** 

Launch this with DeepSpeed accelerate, for example:

```launch.sh
export CUDA_VISIBLE_DEVICES=0,1

num_gpus=2
batch_size_per_gpu=1
total_batch_size=512
num_train_epochs=2
learning_rate=1e-4

## g.t. number of examples in dataset
num_training_examples=???

## recompute gradient accumulation steps and save steps to reflect device count / batch size per device
gradient_acc_steps=$(($total_batch_size/$num_gpus/$batch_size_per_gpu))

## recompute save steps to save at the end of each epoch
save_steps=$(($num_training_examples/$total_batch_size))

## launch, with run-time config overrides
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $num_gpus \
    --use_deepspeed \
    --main_process_port 29500 \
    --deepspeed_config_file experiments/stage3_no_offloading_accelerate.conf \
    experiments/finetuning.py \
    --config experiments/sample_finetuning_config.yaml \
    --per_device_train_batch_size $batch_size_per_gpu \
    --gradient_accumulation_steps $gradient_acc_steps \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --save_steps $save_steps \
    --output_dir ??? \
```

@misc{wang2023far,
   title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources}, 
   author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
   year={2023},
   eprint={2306.04751},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}

"""

import argparse
import random
import pathlib
import sys
import yaml
from functools import partial
import os

import torch
from accelerate import Accelerator
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer

from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    DataCollatorForSeq2Seq,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
)
from accelerate.logging import get_logger
from datasets import load_dataset
import wandb

logger = get_logger(__name__)


def save_with_accelerate(accelerator, model, tokenizer, output_dir):
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=False,
    )


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    prompt = example["prompt"]
    example_text = example["completion"] + tokenizer.eos_token

    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )

    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def main(args):
    # read config from a yaml config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for arg in args.__dict__:
        if not arg in config["training_args"]:
            continue
        new_value = args.__dict__[arg]
        if not new_value:
            continue
        print(f"overwriting {arg} to {new_value}")
        config["training_args"][arg] = new_value

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = config.get(
            "wandb",
            {
                "project": "",
                "entity": "",
                "dir": "",
            },
        )
        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["training_args"]["run_name"],
            config=config,
            dir=wandb_kwargs["dir"],
        )

    accelerator.wait_for_everyone()

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    print(f"Number of parameters: {model.num_parameters()}")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
        embedding_size = embeddings.weight.shape[0]
    # load dataset
    data_files = {}
    dataset_args = {}
    data_files["train"] = os.path.join(config["data_dir"], config["train_file"])
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    # hf_datasets["train"] = hf_datasets["train"].select(range(int(config["num_train"])))
    context_length = config["context_length"]

    # Preprocessing the datasets.

    ### Check that all the expected fields are present
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=context_length,
    )

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=4,
            load_from_cache_file=not config["overwrite_cache"],
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(
            lambda example: (example["labels"] != -100).any()
        )

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # prepare training
    training_args = TrainingArguments(**config["training_args"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", return_tensors="pt"
    )

    def sum_loss_fn(outputs, labels, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # reduce loss is sum
        # this ensures that we weight all tokens in the dataset equally,
        # rather than weighting each overall example equally when
        # using high amounts of gradient accumulation.
        # this can result in > 5 point improvements in AlpacaEval
        # see https://github.com/huggingface/transformers/issues/24725 for
        # more discussion and details.
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_loss_func=sum_loss_fn if config["reduce_loss"] == "sum" else None,
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()

    if accelerator.is_main_process:
        tokenizer.save_pretrained(config["training_args"]["output_dir"])
    save_with_accelerate(
        accelerator, model, tokenizer, config["training_args"]["output_dir"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/conf.yaml")

    ### Overrideable arguments: (ugly) patch to enable dynamic recomputation of e.g.
    ### gradient accumulations step at launch time based on available GPU configuration
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=0)

    args = parser.parse_args()

    main(args)
