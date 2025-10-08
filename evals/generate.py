import os
import argparse
from typing import List
import random
import json
import sys
import pathlib
import torch
import pdb
import numpy as np
import functools
from datasets import load_dataset
from vllm import LLM, SamplingParams

BASE_PATH =  str(pathlib.Path().resolve())
sys.path.insert(0, BASE_PATH)
from evals.utils import *

DATASET_LOOKUP = {
	'humanevalsynth': (f'{BASE_PATH}/evals/data/humanevalsynth_compl_aug.jsonl',),
	'mbpp': (f'{BASE_PATH}/evals/data/mbpp_compl_aug.jsonl',),
	'humanevalfix': ("bigcode/humanevalpack", "python"),
	'swebench': ('princeton-nlp/SWE-bench_oracle',)
}

def get_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--run_base_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="try")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Dataset arguments
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # LM arguments
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B"
    )  # model path
    parser.add_argument(
        "--base_model_name", type=str, default="Llama-3.2-1B"
    )  # model name

    parser.add_argument("--no_diffs", type=int, default=0)  # is model trained on diffs?
    parser.add_argument("--base_model", type=int, default=0)  # is this a base model?
    parser.add_argument("--completion_variant", type=int, default=0)  # running the completion variant of a synthesis task?
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_num_seqs", type=int, default=512)

    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)

    args = parser.parse_args()
    return args

def get_formatted_prompt(datum, completion_variant: int = 0, dataset: str = None) -> str:
	def _vanilla_helper(datum, completion_variant):
		filename = f"{datum['entry_point']}_solution.py"
	    if completion_variant and datum["starting_code"] is None: # no starting code -> example where Human solution only has a single line of code in its body
	    	return ""
	    elif completion_variant:
	    	filecontents = add_line_numbers_to_code(datum["starting_code"])
	    else:
	    	filecontents = ""
	    formatted_code_context = CODE_CONTEXT_FORMAT.format(
	        filename=filename, filecontents=filecontents
	    )
	    instruction = datum["instruction"]

	    ## put instruction/issue together with reformated code contents
	    reformatted_prompt = PROMPT_TEMPLATE.format(
	        formatted_code_context=formatted_code_context,
	        instruction=instruction,
	    )
	    return reformatted_prompt

	def _swebench_helper(datum):
		prompt = datum["text"]
		instruction = prompt.split("<issue>")[1].split("</issue>")[0]
	    start_code_state = prompt.split("<code>")[1].split("</code>")[0]

	    ## strip code line number and re-add using our format
	    repository_state = {}
	    current_filename = ""
	    current_file_contents = []

	    for line in start_code_state.split("\n"):
	        if line.startswith("[end of "):
	            program = remove_line_numbers_from_code("\n".join(current_file_contents))
	            repository_state[current_filename] = add_line_numbers_to_code(program)
	            current_file_contents = []
	            current_filename = ""
	            continue

	        if line.startswith("[start of "):
	            current_filename = line.split("[start of ")[1].rstrip("]").rstrip()
	            continue

	        current_file_contents += [line]

	    formatted_code_context = ""
	    for filename, file_contents in repository_state.items():
	        if not ".py" in filename:
	            continue
	        formatted_code_context += (
	            "[start of {filename}]\n{file_contents}\n[end of {filename}]\n".format(
	                filename=filename, file_contents=file_contents
	            )
	        )
	    formatted_code_context = formatted_code_context.rstrip()

	    ## put instruction/issue together with reformated code contents
	    reformatted_prompt = PROMPT_TEMPLATE.format(
	        formatted_code_context=formatted_code_context,
	        instruction=instruction.lstrip().rstrip(),
	    )

	    return reformatted_prompt

	if dataset == 'humanevalfix':
		datum["instruction"] = f'Fix bugs in {datum["entry_point"]}.'
	elif completion_variant == 1 and dataset in ('mbpp', 'humanevalsynth')
		if dataset == 'mbpp':
			datum["instruction"] = f"Complete the Python function {datum["entry_point"]}."
		else:
			fname = datum["signature"].split("(")[0].split(" ")[-1]
    		datum["instruction"] = f"Complete the Python function {fname}."
	else:
		raise ValueError(f"unexpected combination, dataset {dataset} and task variant {completion_variant}")


	if dataset == "swebench":
		formatting_helper_fn = _swebench_helper
	else:
		formatting_helper_fn = functools.partial(_vanilla_helper, completion_variant=completion_variant)

	datum["eval_prompt"] = formatting_helper_fn(datum)
	return datum

def load_and_preprocess_data(dataset, split, completion_variant):
	dataset_args = DATASET_LOOKUP[dataset]
	# try loading from Hub
    try:
    	dataset = load_dataset(*dataset_args, trust_remote_code=True)[split]
    except BaseException as e1:
    	# try loading locally
    	try:
    		data_files = {
    			split: dataset_args[0]
    		}
		    dataset = load_dataset("json", data_files=data_files)[split]
    	except BaseException as e2:
    		raise ValueError(f"Failed to load dataset {dataset} from Hub ({e1}) and locally ({e2})")

    formatted_prompt_fn = functools.partial(get_formatted_prompt, dataset=args.dataset, completion_variant=args.completion_variant)

    dataset = dataset.map(formatted_prompt_fn)

    # Remove any examples where 'text' is empty (for code_contests)
    dataset = dataset.filter(lambda example: example["text"] != "")

def post_process_vllm_output(
    datum, prompt, completion, no_diffs=False, base_model=False
):
    if not no_diffs:
        repository = parse_start_state_of_repository(prompt)
        if not base_model:
            if "</code>" in completion:
                completion = completion[: completion.find("</code>")]
            completion = completion[
                completion.find(f"{DIFF_TOKEN}") + len(DIFF_TOKEN) :
            ]
        filename = [file for file in repository][0]
        return apply_diffs(repository[filename], completion)
    return completion


def main():
    args = get_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check if generation was already run
    if os.path.exists(os.path.join(args.run_dir, f"predictions.json")):
        print("this generation was already conducted! exiting early.")
        return

    # load & preprocess dataset
    print(f"loading dataset")
    dataset = load_and_preprocess_data(args.dataset, args.split)

    # unpack prompts, problem ids, and tests
    prompts = [
        get_formatted_prompt(dataset[i], base_model=args.base_model)
        for i in range(len(dataset))
    ]
    problem_ids = [dataset["task_id"][i] for i in range(len(dataset))]
    tests = [dataset["test"][i] for i in range(len(dataset))]

    if not args.end_idx is None:
        prompts = prompts[args.begin_idx : args.end_idx]
        problem_ids = problem_ids[args.begin_idx : args.end_idx]
        tests = tests[args.begin_idx : args.end_idx]

    print(f"len(prompts): {len(prompts)}")
    print(prompts[0])

    # set up model and sampling params
    llm, tokenizer = setup_model(
        args.model_name,
        args.max_model_len,
        args.trust_remote_code,
        args.num_gpus,
        args.max_num_seqs,
        tokenizer_name=args.tokenizer_name,
    )
    sampling_params = SamplingParams(
        stop_token_ids=[],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.num_samples,
        top_p=args.top_p,
        min_p=args.min_p,
    )

    post_process_fn = post_process_vllm_output

    responses = llm.generate(prompts, sampling_params)

    ## save parsed outputs
    sampling_outputs = {
        problem_ids[r]: [
            post_process_fn(dataset[r], output.text) for output in response.outputs
        ]
        for r, response in enumerate(responses)
    }

    tests_by_pid = {
        problem_ids[i]: tests[i] for i in range(args.end_idx - args.begin_idx)
    }
    sampling_outputs["tests_by_pid"] = tests_by_pid

    token_counts = {}
    for r, response in enumerate(responses):
        token_counts[f"{problem_ids[r]} prompt"] = len(response.prompt_token_ids)
        token_counts[f"{problem_ids[r]} outputs"] = [
            len(output.token_ids) for output in response.outputs
        ]

    with open(os.path.join(args.run_dir, f"predictions.json"), "w") as f:
        json.dump(sampling_outputs, f)

    with open(os.path.join(args.run_dir, f"token_counts.json"), "w") as f:
        json.dump(token_counts, f)

    ### Also save unprocessed (i.e. "raw") outputs to file
    sampling_raw_outputs = {
        problem_ids[r]: [output.text for output in response.outputs]
        for r, response in enumerate(responses)
    }

    with open(os.path.join(args.run_dir, f"raw_predictions.json"), "w") as f:
        json.dump(sampling_raw_outputs, f)


if __name__ == "__main__":
    main()
