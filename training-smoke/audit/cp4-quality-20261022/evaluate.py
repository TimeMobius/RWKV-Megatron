"""Paired native-BF16 quality evaluation of initial and 12-hour RWKV weights."""

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training-smoke/audit'))

from megatron.core import parallel_state as ps
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model
from transformers import AutoTokenizer

VALID_VOCAB = 248077
BENCHMARKS = ROOT / 'training-smoke/qwen-rwkv-qwenvocab/common-benchmarks-20261021/samples.json'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def freeze(cfg):
    cache = ROOT / 'training-smoke/audit/cp4-longrun-20261021/data-cache'
    train = np.load(next(cache.glob('*-train-document_index.npy')))
    valid = np.load(next(cache.glob('*-valid-document_index.npy')))
    assert not np.intersect1d(train, valid).size
    indices = np.load(next(cache.glob('*-valid-sample_index.npy')))
    shuffle = np.load(next(cache.glob('*-valid-shuffle_index.npy')))
    used = set()
    for index in shuffle[:32]:
        begin, _ = indices[index]
        end, _ = indices[index + 1]
        used.update(map(int, valid[begin:end + 1]))
    rng = np.random.default_rng(20261022)
    source = IndexedDataset(cfg['data_path'][0])
    samples = []
    for doc in rng.permutation(np.unique(valid)):
        length = int(source.sequence_lengths[doc])
        if int(doc) in used or length < 513:
            continue
        offset = int(rng.integers(length - 513 + 1))
        ids = source.get(int(doc), offset=offset, length=513).copy().astype(np.int64).tolist()
        assert min(ids) >= 0 and max(ids) < VALID_VOCAB
        samples.append(dict(document=int(doc), offset=offset, token_ids=ids))
        if len(samples) == 256:
            break
    assert len(samples) == 256
    save(HERE / 'heldout-samples.json', dict(
        seed=20261022, samples=samples, train_valid_document_ids_disjoint=True,
        excluded_fixed_validation_documents=len(used), length=513,
        benchmark_samples_sha256=hashlib.sha256(BENCHMARKS.read_bytes()).hexdigest()))


def build_model(cfg, tp=1):
    d = cfg['language_model']
    c = TransformerConfig(
        num_layers=d['num_layers'], hidden_size=d['hidden_size'],
        num_attention_heads=d['num_attention_heads'], ffn_hidden_size=d['ffn_hidden_size'],
        kv_channels=64, tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
        sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16, hidden_dropout=0., attention_dropout=0.,
        add_bias_linear=False, add_qkv_bias=False, perform_initialization=False,
        gradient_accumulation_fusion=False, bias_dropout_fusion=True,
        persist_layer_norm=False, layernorm_epsilon=1e-5, enable_autocast=False)
    for key in ['D_AAA_LORA', 'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA']:
        setattr(c, key, d[key])
    c.rwkv_key_fusion = True
    model = RWKV7Model(c, transformer_layer_spec=get_rwkv7_layer_spec(), vocab_size=248320,
                      max_sequence_length=1024, parallel_output=False,
                      share_embeddings_and_output_weights=False,
                      position_embedding_type='none').cuda().bfloat16()
    model.eval().requires_grad_(False)
    return model


def forward(model, ids):
    tokens = torch.tensor([ids], dtype=torch.long, device='cuda')
    positions = torch.arange(len(ids), device='cuda')[None]
    return model(tokens, positions, None)


def benchmarks(model, report):
    samples = json.loads(BENCHMARKS.read_text())
    for task, data in samples.items():
        rows = []
        for index, sample in enumerate(data['samples']):
            likelihoods = []
            exact = False
            for req in sample['requests']:
                output = forward(model, req['input_ids'])
                start = req['score_start']
                logits = output[0, start:start + len(req['target_ids']), :VALID_VOCAB].float()
                targets = torch.tensor(req['target_ids'], device='cuda')
                assert len(logits) == len(targets) and torch.isfinite(logits).all()
                ce = F.cross_entropy(logits, targets, reduction='none')
                likelihoods.append(-float(ce.sum()))
                exact = bool((logits.argmax(-1) == targets).all())
                del logits, output
            normalized = [ll / req['normalization_chars']
                          for ll, req in zip(likelihoods, sample['requests'])]
            row = dict(source_index=sample['source_index'], gold=sample['gold'],
                       loglikelihoods=likelihoods, pred=int(np.argmax(likelihoods)),
                       pred_norm=int(np.argmax(normalized)))
            row.update(correct=row['pred'] == row['gold'],
                       correct_norm=row['pred_norm'] == row['gold'])
            if task == 'lambada_openai':
                row['exact_match'] = exact
            rows.append(row)
            if (index + 1) % 25 == 0:
                print(task, index + 1, flush=True)
        if task == 'lambada_openai':
            metrics = dict(acc=float(np.mean([r['exact_match'] for r in rows])),
                           last_word_ppl=math.exp(-float(np.mean([r['loglikelihoods'][0] for r in rows]))))
        else:
            metrics = dict(acc=float(np.mean([r['correct'] for r in rows])),
                           acc_norm=float(np.mean([r['correct_norm'] for r in rows])))
        report['benchmarks'][task] = dict(n=len(rows), metrics=metrics, per_sample=rows)
        print('RESULT', task, metrics, flush=True)
        save(HERE / f"{report['model']}-results.json", report)


def heldout(model, report):
    samples = json.loads((HERE / 'heldout-samples.json').read_text())['samples']
    rows = []
    for i, sample in enumerate(samples):
        output = forward(model, sample['token_ids'][:-1])
        logits = output[0, :, :VALID_VOCAB].float()
        targets = torch.tensor(sample['token_ids'][1:], device='cuda')
        assert torch.isfinite(logits).all()
        ce = F.cross_entropy(logits, targets, reduction='none')
        rows.append(dict(document=sample['document'], ce=float(ce.mean()),
                         token_accuracy=float((logits.argmax(-1) == targets).float().mean())))
        if i == 0:
            alternative = torch.logsumexp(logits, -1) - logits.gather(-1, targets[:, None]).squeeze(-1)
            assert float((alternative - ce).abs().max()) < 1e-5
            assert torch.equal(forward(model, sample['token_ids'][:-1]), output)
            report['checks'].update(repeated_forward_exact=True, ce_formula_verified=True)
            torch.save(output[:, :64].cpu(), HERE / f"{report['model']}-calibration.pt")
        if (i + 1) % 64 == 0:
            print('heldout', i + 1, flush=True)
        del output, logits, ce
    ce = float(np.mean([r['ce'] for r in rows]))
    report['heldout'] = dict(n=256, predicted_tokens=256 * 512, ce=ce, ppl=math.exp(ce),
                             token_accuracy=float(np.mean([r['token_accuracy'] for r in rows])),
                             per_sample=rows)


def generations(model, cfg, report):
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_model'], trust_remote_code=True,
                                             local_files_only=True)
    prompts = [
        ('zh_continuation', '人工智能正在改变我们的生活。例如，在医疗领域，'),
        ('en_continuation', 'The most important lesson I learned from this experience was'),
        ('code', 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n'),
        ('zh_qa', '问：中国的首都是哪里？\n答：'),
        ('arithmetic', 'Question: What is 17 + 25?\nAnswer:'),
        ('translation', '将下面这句话翻译成英文：今天天气很好，我们去公园散步。\n译文：'),
        ('summary', '请用一句话概括：小李每天骑自行车上班，既节省了交通费用，也锻炼了身体。\n摘要：'),
        ('chat', tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '请解释什么是上下文并行，回答不超过三句话。'}],
            tokenize=False, add_generation_prompt=True)),
    ]
    for i, sample in enumerate(json.loads((HERE / 'heldout-samples.json').read_text())['samples'][:2]):
        prompts.append((f'heldout_{i}', tokenizer.decode(sample['token_ids'][:64])))
    outputs = []
    select = {'position': 0}

    def head_input(_module, args):
        return (args[0][select['position']:select['position'] + 1], *args[1:])

    for name, prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) + 128 <= 256
        tokens = torch.zeros((1, 256), dtype=torch.long, device='cuda')
        positions = torch.arange(256, device='cuda')[None]
        tokens[0, :len(ids)] = torch.tensor(ids, device='cuda')
        select['position'] = len(ids) - 1
        if not outputs:
            expected = model(tokens, positions, None)[:, select['position'], :VALID_VOCAB]
        handle = model.output_layer.register_forward_pre_hook(head_input)
        generated = []
        try:
            for j in range(128):
                select['position'] = len(ids) + j - 1
                logits = model(tokens, positions, None)[0, 0, :VALID_VOCAB]
                if not outputs and not j:
                    assert torch.equal(logits, expected[0])
                    report['checks']['selected_head_matches_full_logits'] = True
                assert torch.isfinite(logits).all()
                token = int(logits.argmax())
                generated.append(token)
                tokens[0, len(ids) + j] = token
                if token in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')}:
                    break
        finally:
            handle.remove()
        outputs.append(dict(name=name, prompt=prompt, generated=tokenizer.decode(generated),
                            token_ids=generated, decoding='greedy, max_new_tokens=128, no penalties'))
        print('GEN', name, outputs[-1]['generated'], flush=True)
        report['generations'] = outputs
        save(HERE / f"{report['model']}-results.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['initial', 'final'], required=True)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / 'training-smoke/audit/cp4-longrun-20261021/config-train.yaml').read_text())
    if args.freeze_only:
        freeze(cfg)
        return
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch._dynamo.config.cache_size_limit = 256
    dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl='local')
    model = build_model(cfg)
    path = Path(cfg['load_pretrained_rwkv']) if args.model == 'initial' else HERE / 'export/iter_0049189/rwkv_state_dict.pth'
    report = dict(model=args.model, weights=str(path), benchmarks={}, checks={},
                  valid_vocab_size=VALID_VOCAB, precision='BF16 native Megatron/FLA, TP1/CP1',
                  benchmark_samples_sha256=hashlib.sha256(BENCHMARKS.read_bytes()).hexdigest(),
                  heldout_samples_sha256=hashlib.sha256((HERE / 'heldout-samples.json').read_bytes()).hexdigest())
    started = time.monotonic()
    with torch.inference_mode():
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        model.from_rwkv_lm(state, False)
        del state
        gc.collect()
        heldout(model, report)
        save(HERE / f'{args.model}-results.json', report)
        benchmarks(model, report)
        generations(model, cfg, report)
    report['elapsed_seconds'] = time.monotonic() - started
    save(HERE / f'{args.model}-results.json', report)
    print('DONE', args.model, report['elapsed_seconds'], flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
