import os
import math
import json
import yaml
import time
import copy
import random
import argparse
import functools
from datetime import datetime
from string import ascii_uppercase
import concurrent.futures
import torch
import torch.distributed as dist
import asyncio
import pdb
import sys
import pathlib
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(pathlib.Path().resolve()))
from evals.utils import *


def get_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--run_base_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="try")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Dataset arguments
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="test")

    # LM arguments
    parser.add_argument("--diff", type=int, default=0)
    parser.add_argument("--base_model_name", type=str, default="Llama-3.2-1B")
    # Execution arguments
    parser.add_argument("--eval_mode", type=str, default="public_eval")
    parser.add_argument("--k_list", type=str, default="1,5,10")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=5)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    assert (
        args.dataset != "swebench"
    ), "evaluations of SWE-Bench generation is not supported locally -- use the official SWE-Bench repo"

    # load outputs
    output_file = os.path.join(args.run_dir, "predictions.json")
    out_file = f"{output_file.replace('.json', '_evaluated.json')}"
    if os.path.exists(out_file):
        print("this evaluation was already conducted! exiting early.")
        return

    try:
        with open(output_file, "r") as f:
            predictions = json.load(f)
    except:
        raise Exception(f"Cound not find predictions at {output_file}")

    tests_by_pid = predictions["tests_by_pid"]
    del predictions["tests_by_pid"]

    ## flatten all dict data into nested lists for eval
    problem_ids = [pid for pid in predictions]
    tests = [tests_by_pid[pid] for pid in predictions]
    outputs = [
        [prediction for prediction in predictions[problem_id]]
        for problem_id in problem_ids
    ]
    scores = vanilla_evaluation(
        outputs, tests, num_workers=args.num_workers, timeout=args.timeout
    )

    execution_traces = [
        [score[2].strip() for score in score_list] for score_list in scores
    ]
    grades = [[bool(score[1]) for score in score_list] for score_list in scores]

    total = np.array([len(score_list) for score_list in scores])
    correct = np.array([sum(grade_list) for grade_list in grades])

    ## compute pass at k and dump to file
    ks = args.k_list.split(",")
    if not isinstance(ks, (list, tuple)):
        ks = [ks]

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, int(k)).mean()
        for k in ks
        if (total >= int(k)).all()
    }

    print(pass_at_k)

    metrics_save_dir = os.path.join(args.run_dir, "metrics.json")
    with open(metrics_save_dir, "w") as f:
        json.dump(pass_at_k, f, indent=4)

    new_predictions = []
    for i, problem_name in enumerate(problem_ids):
        outputs = predictions[problem_name]
        d = {"base_solutions": outputs}
        d["execution_traces"] = execution_traces[i]
        d["grades"] = grades[i]
        d["problem_name"] = problem_name
        new_predictions += [d]

    # Save to file
    out_file = f"{output_file.replace('.json', '_evaluated.json')}"
    with open(out_file, "w") as f:
        json.dump(new_predictions, f, indent=4)


if __name__ == "__main__":
    main()
