import pandas as pd
import vllm
import torch
import argparse
import os
import sys
import pathlib
import json
import jsonlines
from tqdm import tqdm
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import apply_diffs

prompt_template = """Evaluate the following Python code extract for its potential usefulness for studying Python programming up to the competitive programming level. 

Use the following **4-point scoring system** described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract **contains some correct code that reflects foundational Python programming concepts**, even if its not very useful for solving hard programming problems.
- Award another point if the extract **correctly demonstrates examples of how to use common Python libraries**.
- Add a third point if the code correctly uses a **more advanced data structure or algorithm**.
- Award a fourth point if the extract reflects Python code that is **outstanding in its educational value for competitive programming**.

The Python code extract:
```
{program}
```

After examining the extract:
- Briefly justify your total score, up to 50 words.
- Conclude with the score using the format: Final score: <total points>."""

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
MAX_TOKENS_TO_GEN = 512


def count_lines_in_file(file_path):
    """Efficiently count the number of lines in a file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def get_prompt(code, tokenizer):
    prompt = prompt_template.format(program=code)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{_MAGIC_SPLITTER_}"},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]


def main(args):
    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        quantization=args.quantization,
        max_num_seqs=64,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_files = [
        os.path.join(args.path_to_data_files, fn)
        for fn in os.listdir(args.path_to_data_files)
    ]
    data_files.sort()

    os.makedirs(args.save_directory, exist_ok=True)

    get_prompt_fn = functools.partial(get_prompt, tokenizer=tokenizer)
    max_seq_length = tokenizer.model_max_length

    print("counting total samples")
    total_samples = sum(count_lines_in_file(data_file) for data_file in data_files)

    with tqdm(total=total_samples, desc="Processing Samples") as total_progress:
        for data_file in tqdm(data_files, desc="Processing Files", position=0):
            df = pd.read_json(data_file, lines=True)
            base = os.path.basename(data_file)
            output_file_path = os.path.join(
                args.save_directory,
                f"{base[:base.rfind('.')]}_llama70bit_scores.jsonl",
            )
            start_idx = 0
            if os.path.exists(output_file_path):
                start_idx = count_lines_in_file(output_file_path)

            for i in range(start_idx, len(df), args.batch_size):
                code_batch = df["text"][i : min(i + args.batch_size, len(df))]
                proc_code_batch = []
                entry_ids = []
                for j, code in enumerate(code_batch):
                    tokenized_input = tokenizer.encode(code)
                    if len(tokenized_input) <= (max_seq_length - MAX_TOKENS_TO_GEN):
                        proc_code_batch += [code]
                        entry_ids += [i + j]
                    else:
                        continue

                prompt_batch = [get_prompt_fn(code) for code in proc_code_batch]
                sampling_params = vllm.SamplingParams(max_tokens=MAX_TOKENS_TO_GEN, n=1)
                generations = model.generate(prompt_batch, sampling_params)
                outputs = [output.text for it in generations for output in it.outputs]
                with open(output_file_path, mode="a") as outfile:
                    for idx, output in enumerate(outputs):
                        entry_index = entry_ids[idx]
                        outline = {"id": entry_index, "llm": output}
                        outline = json.dumps(outline) + "\n"
                        outfile.write(outline)

                total_progress.update(len(code_batch))


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
        default=1024,
    )
    parser.add_argument(
        "--path_to_data_files",
        type=str,
        help="Folder containing (JSONL-formatted) data",
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
