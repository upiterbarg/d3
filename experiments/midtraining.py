"""
** Mid-training code ** 

Arguments / hyperparameters are configured directly in this file.

Before launching, you must convert pretrained Llama model weights into the OLMo-core distributed
format and tokenize your data with the dolma-toolkit.

Launch this with torchrun:

```
torchrun \
    --nproc-per-node=2 \
    experiments/midtraining.py
```

For multi-node training:
```
torchrun \
    --nnodes=$NODES \
    --nproc_per_node=$GPUS \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    experiments/pretraining.py
```

This script executes pretraining/mid-training utilities from: 

@misc{olmo20242olmo2furious,
      title={2 OLMo 2 Furious}, 
      author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656}, 
}
"""

import os
import sys
from dataclasses import dataclass
from typing import List, cast

import torch

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.nn.transformer.config import (
    TransformerActivationCheckpointingConfig,
)
from olmo_core.nn.transformer.model import TransformerActivationCheckpointingMode
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.garbage_collector import GarbageCollectorCallback
from olmo_core.utils import get_default_device, seed_all

### SET BASE & CACHE DIRECTORIES
SCRATCH = "path_to_base_directory"
REL_DATA_DIR = "path_to_data_folder_inside_base"
REL_MODEL_DIR = "path_to_converted_model_folder_inside_base"
REL_TOKENIZER_CONFIG = "path_to_tokenizer_config_inside_base"
WORK_DIR = "path_to_temp_or_cache_directory"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 125


def build_config(run_name: str) -> ExperimentConfig:
    SEQ_LEN = 4096

    DP_CONFIG = TransformerDataParallelConfig(
        name=DataParallelType.ddp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )
    USE_FLASH = True
    DTYPE = DType.bfloat16
    AC_CONFIG = None
    COMPILE = False
    SAVE_INTERVAL = 50

    tokenizer_config = TokenizerConfig.from_file(f"{SCRATCH}/{REL_TOKENIZER_CONFIG}")

    print(tokenizer_config.vocab_size)

    model_config = TransformerConfig.llama3_1B(
        vocab_size=tokenizer_config.vocab_size,
        compile=COMPILE,
        fused_ops=False,
        use_flash=USE_FLASH,
        rope_scaling=RoPEScalingConfig(),
        dp_config=DP_CONFIG,
        dtype=DTYPE,
        ac_config=AC_CONFIG,
    )

    optim_config = AdamWConfig(
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.weight"], opts=dict(weight_decay=0.0)
            )
        ],
    )

    dataset_config = NumpyDatasetConfig.glob(
        os.path.expandvars(
            f"{SCRATCH}/{RELATIVE_DATA_DIR}/part-*.npy",
        ),
        name=NumpyDatasetType.fsl,
        sequence_length=SEQ_LEN,
        max_target_sequence_length=SEQ_LEN * 2,
        generate_doc_lengths=True,
        tokenizer=tokenizer_config,
        work_dir=f"{WORK_DIR}",
        dtype=NumpyDatasetDType.uint32,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * SEQ_LEN,
        seed=345,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=os.path.expandvars(
                f"{SCRATCH}/models/midtraining/d3/{run_name}"
            ),
            rank_microbatch_size=4 * SEQ_LEN,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            compile_loss=True,
            load_key_mapping={
                # For backwards compatibility when loading older checkpoints.
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback(
            "lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100))
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500,
                ephemeral_save_interval=SAVE_INTERVAL,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity="",  ## set entity name here
                project="",  ## set project name here
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "garbage_collector",
            GarbageCollectorCallback(enabled=True, gc_interval=SAVE_INTERVAL),
        )
    )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    )


def main(run_name: str):
    config = build_config(run_name)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    device = get_default_device()

    # Build the world mesh, if needed.
    world_mesh = config.model.build_mesh(device=device)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )

    LLAMA_PATH = os.path.expandvars(f"{SCRATCH}/{REL_MODEL_DIR}")
    # TODO: fix checkpoint resume in trainer when torch.compile is enabled.
    load_model_and_optim_state(LLAMA_PATH, model)

    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, mesh=world_mesh)
    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    ### SET RUN NAME
    run_name = "my_run"

    prepare_training_environment()
    try:
        main(run_name)
    finally:
        teardown_training_environment()
