import pytest

from tools.english_benchmark_data import aggregate, encode_pair, parse_gpus, prediction, transform
from tools.evaluate_english_benchmarks import validate_options


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_prepare_pair_scores_every_continuation_token_without_truncation():
    request = encode_pair(Tokenizer(), "Question:", " an answer", 256, 64)
    assert request["input_ids"][-len(request["targets"]):] != request["targets"]
    assert request["score_start"] + len(request["targets"]) == len(request["input_ids"])
    with pytest.raises(ValueError, match="no truncation"):
        encode_pair(Tokenizer(), "x", "y" * 80, 256, 64)


def test_public_templates_match_expected_fields():
    prompt, choices, gold = transform(
        "piqa", dict(goal="Make tea.", sol1="Boil water.", sol2="Freeze water.", label=0)
    )
    assert prompt == "Question: Make tea.\nAnswer:"
    assert choices == ["Boil water.", "Freeze water."]
    assert gold == 0

    prompt, choices, gold = transform("hellaswag", dict(
        activity_label="Cooking", ctx_a="A person cooks.", ctx_b="they",
        endings=["add [title] salt.", "wait."], label="1",
    ))
    assert prompt == "Cooking: A person cooks. They"
    assert choices == ["add. salt.", "wait."]
    assert gold == 1


def test_choice_prediction_keeps_candidate_order_and_lambada_exact_match():
    requests = [dict(targets=[1], chars=3), dict(targets=[2, 3], chars=2)]
    scores = [dict(logprob_sum=-4, target_tokens=1, greedy_match=False),
              dict(logprob_sum=-3, target_tokens=2, greedy_match=True)]
    row = prediction(7, 1, requests, scores, "piqa")
    assert row["prediction"] == 1
    assert row["correct"] is True

    row = prediction(8, 0, requests[:1], scores[:1], "lambada_openai")
    assert row["exact_match"] is False
    assert row["target_nll"] == 4


def test_aggregate_requires_disjoint_complete_rank_indices():
    def record(index, correct=True):
        return dict(index=index, gold=0, prediction=0 if correct else 1,
                    prediction_norm=0 if correct else 1, correct=correct,
                    correct_norm=correct, scores=[
                        dict(logprob_sum=-1, target_tokens=1, greedy_match=correct),
                        dict(logprob_sum=-2, target_tokens=1, greedy_match=False),
                    ])

    def lambada_record(index):
        row = record(index)
        row.update(exact_match=True, target_nll=1.0, target_tokens=1)
        return row

    parts = []
    for rank in range(2):
        tasks = {
            "lambada_openai": [lambada_record(rank), lambada_record(rank + 2)],
            "piqa": [record(rank), record(rank + 2)],
            "hellaswag": [record(rank), record(rank + 2)],
        }
        parts.append(dict(wikitext2=dict(nll=2.0, tokens=2, correct=1), **tasks))
    metrics, _ = aggregate(parts, dict(lambada_openai=4, piqa=4, hellaswag=4), 4, 2)
    assert metrics["wikitext2"]["tokens"] == 4
    assert metrics["piqa"]["acc"] == 1.0
    parts[0]["piqa"][0]["index"] = 9
    with pytest.raises(ValueError, match="partition"):
        aggregate(parts, dict(lambada_openai=4, piqa=4, hellaswag=4), 4, 2)


def test_gpu_list_and_mode_validation():
    assert parse_gpus("0,1,3") == ["0", "1", "3"]
    with pytest.raises(ValueError):
        parse_gpus("0,0")

    class Args:
        mode, limit, context_length, batch_size, save_every = "full", None, 512, 4, 100
    assert validate_options(Args()) == 0
    Args.limit = 12
    with pytest.raises(ValueError, match="forbidden"):
        validate_options(Args())


def test_actual_scoring_preserves_unsorted_choices_and_ignores_padding():
    torch = pytest.importorskip("torch")
    from tools.evaluate_english_benchmarks import Scorer

    class CausalToyModel:
        def __call__(self, tokens, positions, mask):
            # A deterministic causal model, with unique token/position logits.
            target = (tokens.cumsum(-1) + positions + 1) % 8
            return -((torch.arange(8)[None, None] - target[..., None]).float() ** 2)

    model = CausalToyModel()
    scorer = Scorer(model, 8, batch_size=2, device="cpu")
    requests = [
        dict(input_ids=[1, 2, 3, 4], score_start=1, targets=[3, 4, 5], chars=3),
        dict(input_ids=[1, 2], score_start=1, targets=[6], chars=1),
        dict(input_ids=[4, 1, 2], score_start=1, targets=[2, 3], chars=2),
    ]
    batched = scorer.score_requests(requests)
    single = [scorer.score_requests([request])[0] for request in requests]
    assert batched == single
    assert [r["target_tokens"] for r in batched] == [3, 1, 2]
    for request, result in zip(requests, batched):
        tokens = torch.tensor([request["input_ids"]])
        logits = model(tokens, torch.arange(tokens.shape[1])[None], None)
        values = logits[0, request["score_start"]:]
        expected = -torch.nn.functional.cross_entropy(
            values, torch.tensor(request["targets"]), reduction="sum"
        ).item()
        assert result["logprob_sum"] == pytest.approx(expected)

    # A short final window must score only real next-token labels, not padding.
    wiki = scorer.score_wiki([1, 2, 3, 4])
    assert wiki["tokens"] == 3
    tokens = torch.tensor([[1, 2, 3]])
    logits = model(tokens, torch.arange(3)[None], None)[0]
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 3, 4]), reduction="sum").item()
    assert wiki["nll"] == pytest.approx(expected)


def test_wikitext_windows_cover_every_target_once_with_partial_tail():
    from tools.english_benchmark_data import wiki_windows
    ids = list(range(18))
    windows = list(wiki_windows(ids, 8))
    assert [i for i,_ in windows] == [0, 1, 2]
    assert [len(w) for _,w in windows] == [8, 8, 4]
    assert [token for _,w in windows for token in w[1:]] == ids[1:]


def test_resume_rejects_changed_checkpoint_parallelism_and_inputs(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    import tools.evaluate_english_benchmarks as benchmark

    class FakeTokenizer(Tokenizer):
        def get_vocab(self):
            return {"a": 1, "b": 2}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())))
    checkpoint = tmp_path/"iter_0000001"
    model = checkpoint/"mp_rank_00/model_optim_rng.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    prepared = tmp_path/"input.json"
    prepared.write_text("[]")
    from tools.english_benchmark_data import file_hash
    monkeypatch.setattr(benchmark, "ensure_data", lambda *a: {})
    monkeypatch.setattr(benchmark, "prepare_inputs", lambda *a: {
        "dummy": dict(path=str(prepared), sha256=file_hash(prepared))
    })
    args = benchmark.parser().parse_args([
        "--checkpoint", str(checkpoint), "--tokenizer", "fake",
        "--data-dir", str(tmp_path/"cache"), "--output", str(tmp_path/"result"), "--gpus", "0,1"
    ])
    benchmark.prepare(args, {})
    args.resume = True
    benchmark.prepare(args, {})
    args.gpus = "0"
    with pytest.raises(ValueError, match="Cannot resume"):
        benchmark.prepare(args, {})
    args.gpus = "0,1"
    prepared.write_text("[1]")
    with pytest.raises(ValueError, match="data changed"):
        benchmark.prepare(args, {})
    prepared.write_text("[]")
    model.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Cannot resume"):
        benchmark.prepare(args, {})
