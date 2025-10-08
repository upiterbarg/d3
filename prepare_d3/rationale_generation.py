import pandas as pd
import vllm
import torch
import argparse
import os
import sys
import pdb
import pathlib
import json
import jsonlines
import random
from tqdm import tqdm
import functools
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import apply_diffs

description_template = """You are a software engineer. Describe the Python program below using a single sentence that is:
1. In imperative tense
2. Succinct, clear, and describe the changes exactly
3. Free of first person pronouns
4. Professionally written

The Python program:
```
{program}
```

After examining the program, provide your single sentence description using the format:  This is a Python program that: <my description>.
"""

diff_template = """You are a software engineer adding some functionality and/or documentation to a (possibly empty) Python program.  
Your task is to write a **commit message** that describes the changes that you made that is:

1. In imperative tense
2. Succinct, clear, and describes the changes exactly
3. No longer than one sentence
4. Free of first person pronouns
5. Professionally written

The original Python program:
```
{program_start}
```

The Python program after you made changes to it:
```
{program_end}
```

After comparing the Python programs above, provide your commit message describing the functionality and/or documentation that was added using the format: Commit message: <commit message>.
"""

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
MAX_TOKENS_TO_GEN = 256
DIFF_TOKEN = "<|diff|>"


def count_lines_in_file(file_path):
    """Efficiently count the number of lines in a file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def get_prompt(code, tokenizer):
    prompt = description_template.format(program=code)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{_MAGIC_SPLITTER_}"},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]


def get_diff_prompt(code_1, code_2, tokenizer):
    prompt = diff_template.format(program_start=code_1, program_end=code_2)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{_MAGIC_SPLITTER_}"},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]


def main(args):
    df = pd.read_json(
        args.path_to_data_file,
        lines=True,
    )

    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        quantization=args.quantization,
        max_num_seqs=64,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    os.makedirs(args.save_directory, exist_ok=True)

    get_prompt_fn = functools.partial(get_prompt, tokenizer=tokenizer)
    get_diff_prompt_fn = functools.partial(get_diff_prompt, tokenizer=tokenizer)
    max_seq_length = tokenizer.model_max_length

    diffs = []
    description_output_file_path = os.path.join(
        args.save_directory, "llama70bit_filtered_stack_full_synth_labels.jsonl"
    )
    diffs_output_file_path = os.path.join(
        args.save_directory,
        f"llama70bit_filtered_stack_partial_synth_x{args.n_samples_per_file}_labels.jsonl",
    )

    with tqdm(total=len(df), desc="Processing Samples") as total_progress:
        for i in range(0, len(df), args.batch_size):
            text_batch = df["text"][i : min(i + args.batch_size, len(df))]
            code_batch = [apply_diffs("", text) for text in text_batch]
            proc_code_batch = []
            for code in code_batch:
                tokenized_input = tokenizer.encode(code)
                if len(tokenized_input) >= (max_seq_length - MAX_TOKENS_TO_GEN):
                    pcode = tokenizer.decode(
                        tokenized_input[: max_seq_length - MAX_TOKENS_TO_GEN]
                    )
                    pcode = pcode[: pcode.rfind("\n")]
                    proc_code_batch += [pcode]
                else:
                    proc_code_batch += [code]

            prompt_batch = [get_prompt_fn(code) for code in proc_code_batch]

            diff_prompt_batch = []
            diff_batch = []
            for j, text in enumerate(text_batch):
                diffs = (
                    text.lstrip(f"\n{DIFF_TOKEN}")
                    .rstrip(f"\n{DIFF_TOKEN}")
                    .split(f"\n{DIFF_TOKEN}")
                )
                for _ in range(args.n_samples_per_file):
                    start_weights = np.array(
                        [len(diffs) - 2] + [1] * (len(diffs) - 2)
                    )  ## 50% weight on starting from empty state
                    start_weights = start_weights / start_weights.sum()
                    start_choice = int(
                        random.choices(
                            np.arange(len(diffs) - 1), weights=start_weights, k=1
                        )[0]
                    )
                    end_choice = int(
                        random.choice(np.arange(start_choice + 1, len(diffs)))
                    )
                    idx = i + j

                    start_diff_seq = (
                        apply_diffs("", f"\n\n{DIFF_TOKEN}".join(diffs[:start_choice]))
                        if start_choice > 0
                        else ""
                    )
                    end_diff_seq = apply_diffs(
                        "", f"\n{DIFF_TOKEN}".join(diffs[:end_choice])
                    )
                    diff_prompt_batch += [
                        get_diff_prompt_fn(start_diff_seq, end_diff_seq)
                    ]
                    diff_batch += [
                        {
                            "program_start": start_diff_seq,
                            "program_end": end_diff_seq,
                            "metadata": df["metadata"][idx],
                            "id": df["id"][idx],
                            "index": int(df["index"][idx]),
                            "text": f"\n{DIFF_TOKEN}".join(
                                diffs[start_choice:end_choice]
                            ),
                        }
                    ]

            sampling_params = vllm.SamplingParams(max_tokens=MAX_TOKENS_TO_GEN, n=1)
            gen_descriptions = model.generate(prompt_batch, sampling_params)
            gen_descriptions = [
                output.text for it in gen_descriptions for output in it.outputs
            ]

            with open(description_output_file_path, mode="a") as outfile:
                for idx, output in enumerate(gen_descriptions):
                    entry_index = i + idx
                    outline = {"id": entry_index, "llm": output}
                    outline = json.dumps(outline) + "\n"
                    outfile.write(outline)

            gen_diff_descriptions = model.generate(diff_prompt_batch, sampling_params)
            gen_diff_descriptions = [
                output.text for it in gen_diff_descriptions for output in it.outputs
            ]

            with open(diffs_output_file_path, mode="a") as outfile:
                for diff_metadata, diff_description in zip(
                    diff_batch, gen_diff_descriptions
                ):
                    outline = {**diff_metadata, "llm": diff_description}
                    outline = json.dumps(outline) + "\n"
                    outfile.write(outline)

            total_progress.update(len(text_batch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--n_samples_per_file",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--path_to_data_file",
        type=str,
        help="JSONL formatted file containing full edit sequences",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        help="Directory to save the JSON output files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
