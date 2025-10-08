import os
import pathlib
import sys
import pdb
import datasets
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import apply_insertion_diff_seq
from evals.create_compl_task_variant.sampling_utils import *

dataset = datasets.load_dataset("bigcode/humanevalpack")["test"]
samples = []


with tqdm(total=len(dataset)) as pbar:
    for i in range(len(dataset)):
        solution = dataset["prompt"][i] + dataset["canonical_solution"][i]

        edit_sequence = random_chunked_trajectory(solution, ignore_comments=True)

        if edit_sequence is None:
            starting_code = None
        else:
            _, diff_seq = inflate_edit_path(solution, edit_sequence)

            starting_code = apply_insertion_diff_seq(
                f"\n{DIFF_TOKEN}".join(diff_seq[:-1]) + f"\n{DIFF_TOKEN}\n{DIFF_TOKEN}"
            )

        samples += [
            {
                "task_id": dataset["task_id"][i],
                "example_id": j,
                "test": dataset["test"][i],
                "entry_point": dataset["entry_point"][i],
                "starting_code": starting_code,
                "signature": dataset["signature"][i],
                "instruction": dataset["instruction"],
            }
        ]
        pbar.update(1)

df = pd.DataFrame(samples)
os.makedirs("evals/data", exist_ok=True)
df.to_json("evals/data/humanevalsynth_compl_aug.jsonl", orient="records", lines=True)
