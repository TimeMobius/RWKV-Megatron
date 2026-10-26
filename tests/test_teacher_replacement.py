"""Replacing a rejected source window preserves logical sample order and replay."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rwkv_megatron.online_teacher import TeacherSampleError, TeacherTopK
from rwkv_megatron.teacher_replacement import TeacherValidatedDataset


class Source:
    def __len__(self):
        return 6

    def get_source_sample(self, index):
        return dict(tokens=torch.tensor([index, index + 1]), raw_prompts=str(index))


class Teacher:
    prompt_mode = "text"

    def __init__(self):
        self.calls = []

    def get_logits(self, tokens, **kwargs):
        index = int(tokens[0, 0])
        self.calls.append(index)
        if index in (0, 2, 3):
            raise TeacherSampleError("Teacher returned mismatched positions")
        return TeacherTopK(tokens[..., None], torch.zeros(*tokens.shape, 1))


def test_next_valid_sample_replaces_bad_requests_without_duplicate_or_missing_good_rows(
    tmp_path,
):
    teacher = Teacher()
    dataset = TeacherValidatedDataset(Source(), teacher, tmp_path, "same", lookahead=3)
    try:
        actual = [int(dataset[i]["tokens"][0]) for i in range(6)]
        assert actual == [1, 4, 5, 6, 7, 8]
        assert len(actual) == len(set(actual))
        assert dataset.rejected_count() == 3
        records = [
            json.loads(line)
            for line in (tmp_path / "rejected-samples.jsonl").read_text().splitlines()
        ]
        assert sorted(r["source_index"] for r in records) == [0, 2, 3]
        assert all(
            r["reason"] and r["sample_sha256"] and r["token_ids"] for r in records
        )
    finally:
        dataset.close()
    assert all(teacher.calls.count(i) == 1 for i in teacher.calls)


def test_two_pipeline_readers_and_resume_share_selection_and_decisions(tmp_path):
    a = TeacherValidatedDataset(Source(), Teacher(), tmp_path, "same", lookahead=2)
    b = TeacherValidatedDataset(Source(), Teacher(), tmp_path, "same", lookahead=2)

    def read(dataset):
        return [int(dataset[i]["tokens"][0]) for i in range(6)]

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(read, [a, b]))
        assert first == second == [1, 4, 5, 6, 7, 8]
    finally:
        a.close()
        b.close()
    teacher = Teacher()
    resumed = TeacherValidatedDataset(Source(), teacher, tmp_path, "same", lookahead=1)
    try:
        assert int(resumed[4]["tokens"][0]) == 7
        assert teacher.calls == [
            7
        ], "Resume only requests the current sample, not its consumed prefix"
    finally:
        resumed.close()


def test_unrelated_error_does_not_skip_arbitrary_training_data(tmp_path):
    class Broken:
        def get_logits(self, *args, **kwargs):
            raise TimeoutError("service offline")

    dataset = TeacherValidatedDataset(Source(), Broken(), tmp_path, "same", lookahead=1)
    try:
        with pytest.raises(TimeoutError, match="offline"):
            dataset[0]
        assert dataset.rejected_count() == 0
    finally:
        dataset.close()


def test_consecutive_bad_sample_limit_avoids_training_on_no_data(tmp_path):
    class Broken:
        def get_logits(self, *args, **kwargs):
            raise TeacherSampleError("bad tokens")

    dataset = TeacherValidatedDataset(Source(), Broken(), tmp_path, "same", lookahead=1)
    try:
        with pytest.raises(RuntimeError, match="100 consecutive"):
            dataset[0]
    finally:
        dataset.close()


def test_validated_targets_are_paired_and_never_requested_again_by_prefetch(tmp_path):
    from torch.utils.data import DataLoader

    from rwkv_megatron.online_teacher import TeacherPrefetchIterator

    dataset = TeacherValidatedDataset(
        Source(), Teacher(), tmp_path, "paired", lookahead=2
    )

    class NoSecondRequest:
        def get_logits(self, *args, **kwargs):
            pytest.fail("Validated targets must travel with inputs")

    iterator = TeacherPrefetchIterator(
        DataLoader(dataset, batch_size=2, num_workers=0),
        NoSecondRequest(),
        num_workers=1,
        lookahead=2,
    )
    try:
        for batch in iterator:
            targets = batch[iterator.future_key].result()
            assert torch.equal(targets.token_ids[..., 0], batch["tokens"])
        assert iterator.get_stats()["completed"] == 3
    finally:
        iterator.close()
        dataset.close()
