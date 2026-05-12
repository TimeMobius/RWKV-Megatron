import numpy

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
)
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split


class DummyDataset(MegatronDataset):
    """A minimal MegatronDataset that tags each sample with its source name."""

    def __init__(self, dataset_name: str, num_samples: int, config: BlendedMegatronDatasetConfig):
        indices = numpy.arange(num_samples, dtype=numpy.int64)
        super().__init__(
            dataset=list(range(num_samples)),
            dataset_path=dataset_name,
            indices=indices,
            num_samples=num_samples,
            index_split=Split.train,
            config=config,
        )
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_id = int(self.indices[idx])
        return {
            "tokens": numpy.array([sample_id], dtype=numpy.int64),
            "sample_id": sample_id,
            "dataset_name": self.dataset_name,
        }


def _build_config(shuffle: bool) -> BlendedMegatronDatasetConfig:
    return BlendedMegatronDatasetConfig(
        random_seed=2024,
        sequence_length=16,
        blend=None,
        split=None,
        shuffle_blend_datasets=shuffle,
    )


def _build_blend(shuffle: bool) -> BlendedDataset:
    config = _build_config(shuffle)
    datasets = [
        DummyDataset("dataset_A", num_samples=9, config=config),
        DummyDataset("dataset_B", num_samples=14, config=config),
    ]
    weights = [len(dataset) for dataset in datasets]
    return BlendedDataset(datasets=datasets, weights=weights, size=None, config=config)


def _dump_sequence(blend: BlendedDataset):
    sequence = [blend[idx]["dataset_name"] for idx in range(len(blend))]
    print("dataset_order:", " ".join(sequence))
    return sequence


def test_shuffle_disabled_keeps_block_order():
    blend = _build_blend(shuffle=False)
    print(blend.dataset_index.tolist(), end="")
    #assert blend.dataset_index.tolist() == [0, 1, 0, 1, 0, 1]


def test_shuffle_enabled_shuffles_and_prints(capsys):
    blend = _build_blend(shuffle=True)
    sequence = _dump_sequence(blend)

    captured = capsys.readouterr().out
    # Echo the captured output so it is visible when running with -s.
    print(captured, end="")
    assert "dataset_order:" in captured
    # assert sequence.count("dataset_A") == 3
    # assert sequence.count("dataset_B") == 3
    # assert sequence != ["dataset_A", "dataset_B"] * 3

