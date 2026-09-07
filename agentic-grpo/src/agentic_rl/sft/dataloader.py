"""Assembly of the finalized SFT dataset into a PyTorch DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from agentic_rl.sft.collator import DynamicPaddingCollator
from agentic_rl.sft.config import SFTConfig
from agentic_rl.sft.dataset import TokenizedSFTDataset
from agentic_rl.sft.runtime import make_data_generator, seed_data_worker
from agentic_rl.sft.sampler import LengthGroupedBatchSampler


@dataclass(frozen=True)
class SFTDataPipeline:
    dataset: TokenizedSFTDataset
    batch_sampler: Any
    collator: DynamicPaddingCollator
    dataloader: DataLoader[dict[str, Any]]

    def set_epoch(self, epoch: int) -> None:
        """Advance an epoch-aware sampler, if the selected sampler supports it."""

        set_epoch = getattr(self.batch_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)


def build_train_dataloader(
    config: SFTConfig,
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = None,
    verify_hashes: bool = True,
) -> SFTDataPipeline:
    """Build the single-process/single-GPU SFT input pipeline."""

    dataset = TokenizedSFTDataset(
        config.data.path,
        expected_max_seq_length=config.data.max_seq_length,
        verify_hashes=verify_hashes,
    )
    collator = DynamicPaddingCollator(
        pad_token_id=pad_token_id,
        max_seq_length=config.data.max_seq_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    batch_size = config.optimization.micro_batch_size
    if config.data.length_bucket:
        batch_sampler: Any = LengthGroupedBatchSampler(
            dataset.lengths,
            batch_size=batch_size,
            bucket_size=config.data.bucket_size,
            shuffle=config.data.shuffle,
            drop_last=False,
            seed=config.experiment.seed,
        )
    else:
        generator = make_data_generator(config.experiment.seed)
        sample_sampler = (
            RandomSampler(dataset, generator=generator)
            if config.data.shuffle
            else SequentialSampler(dataset)
        )
        batch_sampler = BatchSampler(
            sample_sampler,
            batch_size=batch_size,
            drop_last=False,
        )

    loader_generator = make_data_generator(config.experiment.seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.num_workers > 0,
        worker_init_fn=seed_data_worker,
        generator=loader_generator,
    )
    return SFTDataPipeline(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collator=collator,
        dataloader=dataloader,
    )

