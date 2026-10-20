"""Isolate teacher variability on the actual step-501 input and frozen student.

Run with torchrun, one GPU. Raw request/response evidence remains local in the
output folder; the compact report contains ID-aligned response comparisons.
"""

import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import requests
import torch
import torch.distributed as dist
import yaml

from megatron.core import parallel_state as ps
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars
from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK, kl_divergence_topk
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'teacher-seed-audit-20261020'


def compare_raw(a, b):
    diffs, changed_sets, changed_order = [], 0, 0
    for aa, bb in zip(a['choices'][0]['prompt_logprobs'][1:],
                      b['choices'][0]['prompt_logprobs'][1:]):
        changed_sets += set(aa) != set(bb)
        changed_order += list(aa) != list(bb)
        diffs.extend(abs(aa[k]['logprob'] - bb[k]['logprob']) for k in aa.keys() & bb.keys())
    return dict(changed_token_sets=changed_sets, changed_order=changed_order,
                mean_common_id_logprob_abs=statistics.mean(diffs),
                max_common_id_logprob_abs=max(diffs))


def main():
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-seed1234-600-20261020.yaml').read_text())
    cache = Path(cfg['data_cache_path'])
    selection = json.loads((cache / 'roundtrip-train-selection.json').read_text())
    assert len(selection['selected_indices']) == 600 and selection['rejected_indices'] == [84]
    raw_index = selection['selected_indices'][500]
    assert raw_index == 501
    # Use exactly the cached indices loaded by the continuous and resumed runs.
    dataset = GPTDataset.__new__(GPTDataset)
    dataset.dataset = IndexedDataset(cfg['data_path'][0])
    dataset.config = SimpleNamespace(sequence_length=128, add_extra_token_to_sequence=True)
    prefix = cache / 'fb80eed57e1adeeba21ecd5da5640794-GPTDataset-train'
    for name in ('document_index', 'sample_index', 'shuffle_index'):
        setattr(dataset, name, np.load(str(prefix) + '-' + name + '.npy', mmap_mode='r'))
    text_ids, _, _ = dataset._query_document_sample_shuffle_indices(raw_index)
    tokens = torch.tensor(text_ids[:-1].copy(), dtype=torch.long).unsqueeze(0)
    client = OnlineTeacherClient(cfg['rwkv_teacher_url'], cfg['rwkv_teacher_model'],
                                 cfg['rwkv_teacher_tokenizer'], top_k=100,
                                 api_key='sk-distill', request_seed=1234, cache_size=0)
    prompt = client._decode(tokens[0].tolist())
    assert client._get_tokenizer()(prompt, add_special_tokens=False).input_ids == tokens[0].tolist()
    session = client._get_session()
    payload = dict(model=cfg['rwkv_teacher_model'], prompt=prompt, max_tokens=1,
                   temperature=1.0, top_p=1.0, logprobs=100, prompt_logprobs=100,
                   echo=True, seed=1234)
    records, teachers = [], []
    cases = [('seed1234', 1.0, 1234)] * 4 + [('seed5678', 1.0, 5678)] * 2 + [('greedy', 0.0, 1234)] * 3
    for index, (label, temperature, seed) in enumerate(cases):
        body = json.dumps({**payload, 'temperature': temperature, 'seed': seed},
                          ensure_ascii=False, separators=(',', ':')).encode()
        response = session.post(client.server_url + '/v1/completions', data=body, timeout=120)
        response.raise_for_status()
        result = response.json()
        (OUT / f'request-{index}.json').write_bytes(body)
        (OUT / f'response-{index}.json').write_text(json.dumps(result, ensure_ascii=False))
        ids = torch.full((128, 100), -1, dtype=torch.int32)
        lp = torch.full((128, 100), -torch.inf)
        ids, lp = client._parse_response(result, 128, ids, lp)
        # Confirm every retained (ID, logprob) matches the raw JSON response.
        for pos, values in enumerate(result['choices'][0]['prompt_logprobs']):
            if values is None:
                continue
            for col, key in enumerate(list(values)[:100]):
                assert ids[pos, col] == int(key)
                assert lp[pos, col] == torch.tensor(values[key]['logprob'], dtype=torch.float32)
        teachers.append(TeacherTopK(ids[None, 1:], lp[None, 1:]))
        records.append(dict(label=label, temperature=temperature, seed=seed,
                            request_sha256=hashlib.sha256(body).hexdigest(),
                            response_model=result.get('model'), system_fingerprint=result.get('system_fingerprint'),
                            returned_positions=len(result['choices'][0]['prompt_logprobs']),
                            generated_text=result['choices'][0].get('text'),
                            versus_first=compare_raw(json.loads((OUT / 'response-0.json').read_text()), result)))
        print('teacher response', index, records[-1]['versus_first'], flush=True)

    torch.cuda.set_device(0)
    dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl='local')
    sys.path.insert(0, str(ROOT / 'training-smoke/audit'))
    import distributed as audit
    dims = cfg['language_model']
    audit.FULL = True
    audit.DIMS = dict(layers=dims['num_layers'], hidden=dims['hidden_size'],
                      heads=dims['num_attention_heads'], ffn=dims['ffn_hidden_size'])
    config = audit.config_for(1, 1, False, False)
    for name in ('D_AAA_LORA', 'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA'):
        setattr(config, name, dims[name])
    config.rwkv_key_fusion = True
    model = RWKV7Model(config, transformer_layer_spec=get_rwkv7_layer_spec(),
                      vocab_size=248320, max_sequence_length=128, parallel_output=False,
                      share_embeddings_and_output_weights=False, position_embedding_type='none').cuda().bfloat16()
    checkpoint = Path(cfg['save']) / 'iter_0000500/mp_rank_00/model_optim_rng.pt'
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert saved['iteration'] == 500 and saved['args'].consumed_train_samples == 500
    model.load_state_dict(saved['model'], strict=True)
    model.train()
    tokens = tokens.cuda()
    positions = torch.arange(128, device='cuda').expand_as(tokens)
    with torch.no_grad():
        logits = model(tokens, positions, None)
        other = model(tokens, positions, None)
        forward_equal = torch.equal(logits, other)
        forward_max = float((logits.float() - other.float()).abs().max())
        mask = torch.ones_like(tokens[:, :-1], dtype=torch.bool)
        for row, teacher in zip(records, teachers):
            row['fixed_student_kl'] = float(kl_divergence_topk(logits[:, :-1], teacher, mask,
                                                            temperature=2.0, valid_vocab_size=248077))
        # ID-column permutation must not change the target or KL.
        t = teachers[0]
        permutation = torch.arange(99, -1, -1)
        reordered = TeacherTopK(t.token_ids[..., permutation], t.logprobs[..., permutation])
        reorder_loss = float(kl_divergence_topk(logits[:, :-1], reordered, mask,
                                               temperature=2.0, valid_vocab_size=248077))
    same_body_losses = [row['fixed_student_kl'] for row in records[:4]]
    assert len({row['request_sha256'] for row in records[:4]}) == 1
    greedy_losses = [row['fixed_student_kl'] for row in records if row['label'] == 'greedy']
    report = dict(checkpoint=str(checkpoint), step=501, raw_sample_index=raw_index,
                  token_ids=tokens.cpu().tolist(), parser_matches_raw=True,
                  student_repeated_forward_equal=forward_equal, student_repeated_forward_max_abs=forward_max,
                  fixed_student_same_body_loss_range=max(same_body_losses) - min(same_body_losses),
                  fixed_student_greedy_loss_range=max(greedy_losses) - min(greedy_losses),
                  teacher_column_permutation_loss_abs=abs(reorder_loss - same_body_losses[0]),
                  raw_evidence_directory=str(OUT), records=records,
                  sources=['https://docs.vllm.ai/en/v0.26.0/usage/reproducibility/',
                           'https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/'],
                  limitation='Student forward only; does not isolate optimizer/scheduler/RNG checkpoint parity. Response fingerprint reports vllm-0.26.0-tp4-bd0f1f71; startup configuration, actual kernel selection, routing and quantization were not inspected.')
    (OUT.parent / 'teacher-seed-audit-20261020.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'token_ids'}, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
