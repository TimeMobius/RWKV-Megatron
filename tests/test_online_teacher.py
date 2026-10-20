import pytest
import torch
import requests

from rwkv_megatron.online_teacher import OnlineTeacherClient, RoundTripDataset, TeacherTopK, kl_divergence_topk


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, url, json, timeout):
        self.payload = (url, json, timeout)
        return _Response(
            {
                "choices": [
                    {
                        "prompt_logprobs": [
                            None,
                            {
                                "10352": {"decoded_token": " doctor", "logprob": -3.99, "rank": 12},
                                "42": {"decoded_token": "!", "logprob": -4.5, "rank": 13},
                            },
                        ]
                    }
                ]
            }
        )


class _StatusResponse(_Response):
    def __init__(self, status_code, payload=None, headers=None):
        super().__init__(payload or {})
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, url, json, timeout):
        self.calls += 1
        return self.responses.pop(0)


def _valid_response():
    return _StatusResponse(200, {
        "choices": [{"prompt_logprobs": [None, {
            "10352": {"decoded_token": " doctor", "logprob": -3.99, "rank": 12},
        }]}]
    })


def test_prompt_logprobs_parse_and_request_temperature():
    client = OnlineTeacherClient(
        "http://teacher",
        "Qwen3.8-27B",
        "/unused",
        top_k=2,
        request_temperature=1.0,
    )
    session = _Session()
    client._get_session = lambda: session
    token_ids, logprobs = client._request_prompt_logprobs("hello", sequence_length=2)

    assert token_ids.tolist() == [[-1, -1], [10352, 42]]
    assert torch.allclose(logprobs[1], torch.tensor([-3.99, -4.5]))
    assert session.payload[1]["prompt_logprobs"] == 2
    assert session.payload[1]["temperature"] == 1.0
    assert session.payload[1]["echo"] is True


def test_retry_after_rate_limit_uses_backoff_and_returns_metrics(monkeypatch):
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused",
                                top_k=1, max_retries=2, retry_backoff_initial=7.0,
                                retry_backoff_max=30.0, retry_jitter=0.0)
    session = _SequenceSession([_StatusResponse(429, headers={"Retry-After": "3"}), _valid_response()])
    client._get_session = lambda: session
    sleeps = []
    monkeypatch.setattr("rwkv_megatron.online_teacher.time.sleep", sleeps.append)
    token_ids, logprobs, metrics = client._request_prompt_logprobs("hello", 2, return_metrics=True)
    assert session.calls == 2
    assert sleeps == [3.0]
    assert metrics == {"retries": 1, "rate_limits": 1}
    assert token_ids[1, 0].item() == 10352
    assert client.get_stats()["retries"] == 1
    assert client.get_stats()["rate_limits"] == 1


def test_non_retryable_http_error_is_not_retried():
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused", top_k=1,
                                max_retries=3, retry_jitter=0.0)
    session = _SequenceSession([_StatusResponse(401)])
    client._get_session = lambda: session
    with pytest.raises(requests.HTTPError):
        client._request_prompt_logprobs("hello", 2)
    assert session.calls == 1


def test_identical_batch_is_served_from_thread_safe_cache(monkeypatch):
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused", top_k=1, cache_size=2)
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "x" * len(ids)
        def __call__(self, text, **kwargs):
            return type("Encoded", (), {"input_ids": [1, 2]})()
    client._get_tokenizer = lambda: Tokenizer()
    session = _SequenceSession([_valid_response()])
    client._get_session = lambda: session
    tokens = torch.tensor([[1, 2]])
    first = client.get_logits(tokens)
    second = client.get_logits(tokens.clone())
    assert session.calls == 1
    assert client.get_stats()["cache_hits"] == 1
    assert torch.equal(first.token_ids, second.token_ids)
    assert first.token_ids.data_ptr() != second.token_ids.data_ptr()


def test_teacher_api_key_uses_bearer_header_from_environment(monkeypatch):
    monkeypatch.setenv("RWKV_TEACHER_API_KEY", "test-teacher-key")
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused")
    assert client._get_session().headers["Authorization"] == "Bearer test-teacher-key"


def test_teacher_api_key_explicit_override_and_empty(monkeypatch):
    monkeypatch.setenv("RWKV_TEACHER_API_KEY", "env-key")
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused", api_key="explicit-key")
    assert client._get_session().headers["Authorization"] == "Bearer explicit-key"
    client = OnlineTeacherClient("http://teacher", "Qwen3.8-27B", "/unused", api_key="")
    assert "Authorization" not in client._get_session().headers


def test_topk_kl_temperature_mask_and_gradient():
    student = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]]], requires_grad=True)
    teacher = TeacherTopK(
        token_ids=torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32),
        logprobs=torch.log(torch.tensor([[[0.75, 0.25], [0.6, 0.4]]])),
    )
    mask = torch.tensor([[True, False]])
    loss = kl_divergence_topk(student, teacher, mask, temperature=2.0)
    teacher_probs = torch.softmax(torch.log(torch.tensor([0.75, 0.25])) / 2.0, dim=-1)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student[0, 0] / 2.0, dim=-1)[:2],
        teacher_probs,
        reduction="sum",
    ) * 4.0
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0, 1].abs().sum() == 0


def test_incompatible_vocabulary_fails_before_gather():
    teacher = TeacherTopK(torch.tensor([[[100]]]), torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="student vocabulary"):
        kl_divergence_topk(torch.zeros(1, 1, 3), teacher, torch.ones(1, 1))


def test_roundtrip_filter_replaces_invalid_samples_deterministically():
    from types import SimpleNamespace

    class Dataset:
        index_split = "train"
        def __len__(self):
            return 4
        def __getitem__(self, i):
            return {"tokens": torch.tensor([i])}

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return str(ids[0])
        def __call__(self, text, **kwargs):
            value = int(text)
            return SimpleNamespace(input_ids=[99 if value == 1 else value])

    result = RoundTripDataset(Dataset(), Tokenizer(), 3)
    assert result.indices == [0, 2, 3]
    assert result.rejected_indices == [1]
    assert result.index_split == "train"
    assert result[1]["tokens"].tolist() == [2]
    with pytest.raises(ValueError, match="Only 3/4"):
        RoundTripDataset(Dataset(), Tokenizer(), 4)


def test_missing_teacher_rows_have_zero_finite_gradient():
    logits = torch.randn(1, 3, 4, requires_grad=True)
    teacher = TeacherTopK(
        torch.tensor([[[-1, -1], [1, 2], [1, 2]]]),
        torch.tensor([[[-torch.inf, -torch.inf], [-0.3, -1.0], [-torch.inf, -torch.inf]]]),
    )
    loss = kl_divergence_topk(logits, teacher, torch.ones(1, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, [0, 2]].count_nonzero() == 0


def test_topk_matches_dense_kl_value_and_gradient():
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 11, requires_grad=True)
    ids = torch.stack([torch.randperm(11)[:5] for _ in range(8)]).view(2, 4, 5)
    logprobs = torch.log_softmax(torch.randn(2, 4, 5), dim=-1)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    actual = kl_divergence_topk(logits, TeacherTopK(ids, logprobs), mask, temperature=2.0)
    grad_actual, = torch.autograd.grad(actual, logits)

    # Independent dense, double precision oracle with zero mass outside top-k.
    reference_logits = logits.detach().double().requires_grad_()
    dense = torch.zeros_like(reference_logits).scatter_(-1, ids, (logprobs.double() / 2).softmax(-1))
    terms = torch.nn.functional.kl_div((reference_logits / 2).log_softmax(-1), dense, reduction="none")
    expected = terms.sum(-1)[mask].mean() * 4
    grad_expected, = torch.autograd.grad(expected, reference_logits)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(grad_actual.double(), grad_expected, atol=1e-7, rtol=1e-5)


def test_topk_kl_excludes_padded_student_vocab_rows_from_softmax():
    # Qwen's 248077 usable IDs are padded to 248320 for Megatron.  Give the
    # 243 invalid rows very large logits: they must not affect the KL or its
    # gradient when valid_vocab_size is supplied.
    valid_vocab_size = 3
    logits = torch.tensor([[[2.0, 0.0, -1.0, 100.0, 90.0]]], requires_grad=True)
    teacher = TeacherTopK(
        token_ids=torch.tensor([[[0, 1]]], dtype=torch.int32),
        logprobs=torch.log(torch.tensor([[[0.75, 0.25]]])),
    )
    mask = torch.ones(1, 1, dtype=torch.bool)
    actual = kl_divergence_topk(logits, teacher, mask, temperature=1.0,
                                valid_vocab_size=valid_vocab_size)
    reference = kl_divergence_topk(logits[..., :valid_vocab_size], teacher, mask, temperature=1.0)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.equal(logits.grad[..., valid_vocab_size:], torch.zeros_like(logits.grad[..., valid_vocab_size:]))


def test_topk_kl_rejects_teacher_id_outside_effective_vocab():
    teacher = TeacherTopK(torch.tensor([[[3]]]), torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="student vocabulary"):
        kl_divergence_topk(torch.zeros(1, 1, 5), teacher, torch.ones(1, 1), valid_vocab_size=3)
