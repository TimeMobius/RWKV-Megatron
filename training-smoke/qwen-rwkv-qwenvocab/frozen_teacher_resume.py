"""Diagnostic-only record/replay around the unmodified Megatron training path.

RWKV_FROZEN_MODE=record|replay, RWKV_FROZEN_DIR, RWKV_FROZEN_TRACE are required.
Replay fails on a missing response; it never falls back to the teacher API.
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import pretrain_rwkv7 as pretrain
import megatron.training.training as training
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.global_vars import initialize_runtime_services
from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK

MODE = os.environ['RWKV_FROZEN_MODE']
assert MODE in ('record', 'replay')
FOLDER = Path(os.environ['RWKV_FROZEN_DIR'])
TRACE = Path(os.environ['RWKV_FROZEN_TRACE'])
FOLDER.mkdir(parents=True, exist_ok=True)
TRACE.parent.mkdir(parents=True, exist_ok=True)
records = []
current = None


class FrozenTeacher(OnlineTeacherClient):
    def get_logits(self, input_ids, attention_mask=None, prompts=None):
        tokens = input_ids.detach().cpu().long().contiguous()
        mask = (torch.ones_like(tokens, dtype=torch.bool) if attention_mask is None
                else attention_mask.detach().cpu().bool().contiguous())
        signature = dict(shape=list(tokens.shape), prompts=prompts, top_k=self.top_k,
                         seed=self.request_seed, model=self.model_name, temperature=self.request_temperature)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
        digest.update(tokens.numpy().tobytes())
        digest.update(mask.numpy().tobytes())
        key = digest.hexdigest()
        path = FOLDER / (key + '.pt')
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert torch.equal(saved['tokens'], tokens) and torch.equal(saved['mask'], mask)
            result = TeacherTopK(saved['token_ids'], saved['logprobs'])
            source = 'disk'
        else:
            if MODE == 'replay':
                raise FileNotFoundError(f'Frozen teacher response missing: {key}; HTTP forbidden')
            result = super().get_logits(tokens, attention_mask=mask, prompts=prompts)
            torch.save(dict(tokens=tokens, mask=mask, token_ids=result.token_ids,
                            logprobs=result.logprobs), path)
            source = 'http'
        if current is not None:
            current.update(teacher_key=key, teacher_source=source)
        return TeacherTopK(result.token_ids.clone(), result.logprobs.clone())

    def _get_session(self):
        if MODE == 'replay':
            raise AssertionError('Teacher HTTP is forbidden in replay mode')
        return super()._get_session()


original_loss = pretrain.distillation_loss_func
original_step = training.train_step


def audited_loss(loss_mask, labels, output_tensor, telemetry=None):
    if current is not None:
        # Keep full logits to measure actual differences, not rounded stdout.
        logits_path = TRACE.with_name(TRACE.stem + f"-step{current['step']}-logits.pt")
        logits = output_tensor[0].detach().cpu()
        torch.save(logits, logits_path)
        current['logits_path'] = str(logits_path.resolve())
    result = original_loss(loss_mask, labels, output_tensor, telemetry=telemetry)
    if current is not None:
        pair = result[2]['distill/kl_loss']
        current['kl'] = float(pair[0] / pair[1])
    return result


def audited_step(*args, **kwargs):
    global current
    current = dict(step=int(kwargs['iteration']) + 1)
    try:
        result = original_step(*args, **kwargs)
        current.update(grad_norm=float(result[5]), skipped=int(result[1]),
                       losses={k: float(v) for k, v in result[0].items()})
        records.append(current)
        TRACE.write_text(json.dumps(dict(mode=MODE, steps=records), indent=2) + '\n')
        return result
    finally:
        current = None


if __name__ == '__main__':
    pretrain.OnlineTeacherClient = FrozenTeacher
    pretrain.distillation_loss_func = audited_loss
    training.train_step = audited_step
    pretrain.train_valid_test_datasets_provider.is_distributed = True
    with patch('transformers.dynamic_module_utils.resolve_trust_remote_code', return_value=True):
        args = parse_and_validate_args(extra_args_provider=add_rwkv_arguments)
        assert args.global_batch_size == args.micro_batch_size == 1
        assert not args.rwkv_teacher_async_prefetch
        initialize_runtime_services(args)
        pretrain.get_train_iters()
        config = pretrain_cfg_container_from_args(args)
        try:
            pretrain.pretrain(config, pretrain.train_valid_test_datasets_provider,
                              ModelType.encoder_or_decoder, pretrain.forward_step,
                              model_provider=pretrain.model_provider)
        finally:
            for prefetcher in pretrain._teacher_prefetchers:
                prefetcher.close()
