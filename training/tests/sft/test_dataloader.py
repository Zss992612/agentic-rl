from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agentic_rl.sft.config import load_sft_config
from agentic_rl.sft.dataloader import build_train_dataloader

TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs/sft/retail_qwen3_4b_instruct_2507_lora_v1.yaml"
)
MODEL_PATH = Path("/Users/projects/models/Qwen3-4B-Instruct-2507")


def project_config():
    config = load_sft_config(
        CONFIG_PATH,
        environ={"AGENTIC_RL_MODEL_PATH": str(MODEL_PATH)},
    )
    return replace(
        config,
        data=replace(config.data, num_workers=0, pin_memory=False),
        optimization=replace(config.optimization, micro_batch_size=2),
    )


def test_real_length_grouped_dataloader_produces_training_batch() -> None:
    pipeline = build_train_dataloader(
        project_config(),
        pad_token_id=151_643,
        pad_to_multiple_of=8,
        verify_hashes=False,
    )

    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 256
    assert len(pipeline.dataloader) == 128
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert int(batch["attention_mask"].sum()) > 0
    assert int((batch["labels"] != -100).sum()) > 0


def test_epoch_changes_length_grouped_shuffle_reproducibly() -> None:
    pipeline = build_train_dataloader(
        project_config(),
        pad_token_id=151_643,
        verify_hashes=False,
    )

    epoch_zero = list(pipeline.batch_sampler)
    pipeline.set_epoch(1)
    epoch_one = list(pipeline.batch_sampler)

    assert epoch_zero != epoch_one
    assert sorted(sum(epoch_zero, [])) == list(range(256))
    assert sorted(sum(epoch_one, [])) == list(range(256))


def test_non_bucketed_non_shuffled_pipeline_preserves_dataset_order() -> None:
    config = project_config()
    config = replace(
        config,
        data=replace(config.data, length_bucket=False, shuffle=False),
    )
    pipeline = build_train_dataloader(
        config,
        pad_token_id=151_643,
        verify_hashes=False,
    )

    assert next(iter(pipeline.batch_sampler)) == [0, 1]

