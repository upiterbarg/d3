# D3: A Large Dataset for Training LMs to Act Diff-by-Diff


D3 is a large dataset for training LMs to iteratively synthesize general-purpose Python source code by generating file diffs.
To construct D3, we filter, augment, and annotate source code from The Stack by sampling synthetic file-diff sequences with a code analysis tool (a linter), and labeling each sample with an instruction using an LLM (Llama 3.1 70B Instruct). 

**The full dataset, and the preparation pipeline to reproduce it, will be fully open-sourced if our paper submission is accepted.**

Here, we provide an implementation of core procedures, experiments, and analyses discussed in the submission.

```
prepare_d3/                     # Preparing D3 (note: diff sampling (Phase II-A) is done using pylintseq, https://pypi.org/project/pylintseq/)        
	source_file_grading.py                  # Phase I-C: Grading source files in The Stack for quality & content using Llama 
	rationale_generation.py                 # Phase II-B: Sampling synthetic diff 'sub-trajectories' & labeling all sub- and full- gen. trajectories
experiments/                    # Dual-Stage Training Experiments
	sample_finetuning_config.yaml           # Configuring SFT hyperparameters
	stage3_no_offloading_accelerate.conf    # Configurating DeepSpeed Stage 3
	finetuning.py                           # Core SFT code
	launch_midtraining_pretokinization.sh   # Sample launch script for pretokenizing data for mid-training with Dolma toolkit
	midtraining.py                          # Core mid-training code
evals/                          # Benchmark Evals
	create_compl_task_variant/              # Generation code for procedurally creating the 'completion' synthesis task
		humanevalsynth.py                              # HumanEvalSynth
		mbpp.py                                        # MBPP
		sampling_utils.py                              # Sampling utilities
	evaluate.py                             # Run eval
	generate.py                             # Run generation
	utils.py                                # All utilities (bulk of the supporting code is here)
gemini_topic_discovery.py        # Analyzing topics in D3 by processing instructions with Gemini
```