import os
import pathlib
import sys
import pdb
import datasets
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import apply_insertion_diff_seq, DIFF_TOKEN
from evals.create_compl_task_variant.sampling_utils import *

dataset = datasets.load_dataset("evalplus/mbppplus")["test"]
samples = []


def get_entry_point_and_formatted_nl_instruction(datum):
    def _deconjugate_verb(verb: str) -> str:
        irregular = {
            "is": "be",
            "has": "have",
            "does": "do",
            "goes": "go",
            "says": "say",
        }

        if verb in irregular:
            return irregular[verb]

        if verb.endswith("ies") and len(verb) > 3:
            return verb[:-3] + "y"

        if (
            verb.endswith("es")
            and verb[-3:-2] in "sxz"
            or verb.endswith(("ches", "shes"))
        ):
            return verb[:-2]

        if verb.endswith("s") and len(verb) > 1:
            return verb[:-1]

        return verb  # Return as-is if no transformation is applicable

    functions = [fdef.split("(")[0] for fdef in solution.split("def ")[1:]]
    functions = [
        function if not function.endswith(" ") else function.rstrip()
        for function in functions
    ]

    try:
        entry_point = [
            function for function in functions if function in datum["test_list"][0]
        ][0]
    except BaseException as e:
        raise ValueError(f"Failed to unpack entry point: {e}")

    fdef = solution.split(f"def {entry_point}")[1].split(":")[0]

    nl_problem_desc = ""
    prompt = datum["prompt"].replace(" to that ", " that ")
    if "function to " in prompt:
        prompt = prompt.split("function to ")[-1].capitalize().split(" ", 1)
        nl_problem_desc = " ".join([prompt[0]] + prompt[1:])
    elif "function that ":
        prompt = prompt.split("function that ")[-1].capitalize().split(" ", 1)
        nl_problem_desc = " ".join([_deconjugate_verb(prompt[0])] + prompt[1:])
    else:
        raise ValueError(f"Unexpected prompt structure: {prompt}")

    instruction = f"Write a Python function `{entry_point}{fdef}` to solve the following problem: {nl_problem_desc}"
    return instruction, entry_point


with tqdm(total=len(dataset)) as pbar:
    for i in range(len(dataset)):
        solution = dataset["code"][i]

        edit_sequence = random_chunked_trajectory(solution, ignore_comments=True)

        if edit_sequence is None:
            starting_code = None
        else:
            _, diff_seq = inflate_edit_path(solution, edit_sequence)

            starting_code = apply_insertion_diff_seq(
                f"\n{DIFF_TOKEN}".join(diff_seq[:-1]) + f"\n{DIFF_TOKEN}\n{DIFF_TOKEN}"
            )

        instruction, entry_point = get_entry_point_and_formatted_nl_instruction(
            dataset[i]
        )

        samples += [
            {
                "task_id": dataset["task_id"][i],
                "test": dataset["test"][i],
                "starting_code": starting_code,
                "instruction": instruction,
                "entry_point": entry_point,
            }
        ]
        pbar.update(1)

df = pd.DataFrame(samples)
os.makedirs("evals/data", exist_ok=True)
df.to_json("evals/data/mbpp_compl_aug.jsonl", orient="records", lines=True)
