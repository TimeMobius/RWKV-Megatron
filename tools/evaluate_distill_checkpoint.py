"""Evaluate production RWKV checkpoints on frozen tokens and greedy prompts, without a teacher.

Run via torchrun --standalone --nproc-per-node=1. TP/PP torch checkpoints are
merged with the existing validated converter; no optimizer is loaded or stepped.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROMPTS = (
    ('english', 'The capital of France is'),
    ('chinese', '太阳系中最大的行星是'),
    ('code', 'def add(a, b):\n    return'),
    ('prose', 'The purpose of education is to'),
    ('arithmetic', '17 + 25 ='),
    ('explanation', '光合作用是'),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def fixed_manifest(path, settings, tokenizer, count=32):
    from rwkv_megatron.jsonl_dataset import JsonlTextDataset

    paths = JsonlTextDataset.paths_for_split(settings['data_path'], 'validation')
    identity = dict(protocol=1, tokenizer_class=type(tokenizer).__name__,
                    tokenizer_use_fast=settings.get('rwkv_teacher_tokenizer_use_fast', True),
                    vocabulary_sha256=digest(tokenizer.get_vocab()),
                    valid_vocab=max(tokenizer.get_vocab().values()) + 1,
                    sequence_length=settings['seq_length'], samples=count,
                    chunk_chars=settings.get('rwkv_jsonl_chunk_chars', 512),
                    text_field=settings.get('rwkv_jsonl_text_field', 'text'),
                    sources=[dict(path=str(p.resolve()), sha256=file_hash(p)) for p in paths])
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
        fingerprint = manifest.pop('sha256')
        if digest(manifest) != fingerprint or manifest['identity'] != identity:
            raise ValueError('Fixed validation manifest changed or does not match tokenizer/data/config')
        manifest['sha256'] = fingerprint
        return manifest
    if count < 1:
        raise ValueError('Quality evaluation requires at least one validation sample')
    source = JsonlTextDataset(paths, tokenizer, settings['seq_length'], target_samples=None,
                              chunk_chars=identity['chunk_chars'], text_field=identity['text_field'])
    rows = []
    for index in range(count):
        sample = source.get_source_sample(index)
        rows.append(dict(source_index=index, **{key: sample[key].tolist()
                    for key in ('tokens', 'labels', 'loss_mask')}))
    prompts = [dict(name=name, prompt=text,
                    token_ids=tokenizer(text, add_special_tokens=False).input_ids)
               for name, text in PROMPTS]
    manifest = dict(identity=identity, rows=rows, prompts=prompts)
    manifest['sha256'] = digest(manifest)
    write_json(path, manifest)
    return manifest


def score_logits(logits, labels, mask, valid_vocab):
    """Return additive statistics on every valid next-token label, without padding."""
    selected = mask.bool()
    labels = labels[selected]
    if labels.numel() == 0:
        return dict(nll=0.0, tokens=0, correct=0)
    if labels.min() < 0 or labels.max() >= valid_vocab:
        raise ValueError('Validation label outside the effective vocabulary')
    values = logits[selected, :valid_vocab].float()
    losses = F.cross_entropy(values, labels, reduction='none')
    if not torch.isfinite(losses).all():
        raise ValueError('Non-finite validation NLL')
    return dict(nll=losses.double().sum().item(), tokens=labels.numel(),
                correct=(values.argmax(-1) == labels).sum().item())


def aggregate(rows):
    tokens = sum(row['tokens'] for row in rows)
    if tokens < 1:
        raise ValueError('No effective validation tokens')
    ce = sum(row['nll'] for row in rows) / tokens
    if not math.isfinite(ce):
        raise ValueError('Non-finite validation CE')
    return dict(ce=ce, ppl=math.exp(ce), tokens=tokens, samples=len(rows),
                top1_accuracy=sum(row['correct'] for row in rows) / tokens)


def compare(initial, trained):
    if initial['manifest_sha256'] != trained['manifest_sha256']:
        raise ValueError('Cannot compare checkpoints evaluated on different fixed samples')
    a, b = initial['metrics'], trained['metrics']
    return dict(ce_change=b['ce'] - a['ce'], ppl_ratio=b['ppl'] / a['ppl'],
                top1_accuracy_change=b['top1_accuracy'] - a['top1_accuracy'],
                generations=[dict(name=x['name'], prompt=x['prompt'],
                                  initial=x['continuation'], checkpoint=y['continuation'])
                             for x, y in zip(initial['generations'], trained['generations'])])


def upload_quality_report(path, settings):
    """Publish a separate evaluation run; preserve training run histories."""
    import wandb

    path = Path(path).resolve()
    report = json.loads(path.read_text())
    with wandb.init(entity=settings.get('wandb_entity'),
                    project=settings.get('wandb_project') or 'rwkv-qwen-distill-2p7b',
                    id=uuid.uuid4().hex[:8], resume='never', mode='online',
                    name=path.parent.name + '-checkpoint-quality', job_type='evaluation',
                    dir=str(path.parent),
                    config=dict(checkpoint=report['checkpoint'], iteration=report['iteration'],
                                manifest_sha256=report['manifest_sha256'], teacher_requests=0,
                                optimizer_steps=0)) as run:
        for key in ('ce', 'ppl', 'tokens', 'top1_accuracy'):
            run.summary[f'initial/{key}'] = report['initial']['metrics'][key]
            run.summary[f'checkpoint/{key}'] = report['checkpoint_result']['metrics'][key]
        run.summary['comparison/ce_change'] = report['comparison']['ce_change']
        run.summary['comparison/ppl_ratio'] = report['comparison']['ppl_ratio']
        table = wandb.Table(columns=['prompt', 'initial', 'checkpoint'])
        for row in report['comparison']['generations']:
            table.add_data(row['prompt'], row['initial'], row['checkpoint'])
        run.log({'fixed_prompt_generations': table})
        artifact = wandb.Artifact('rwkv-fixed-checkpoint-quality', type='evaluation')
        artifact.add_file(str(path), name='quality.json')
        artifact.add_file(report['manifest'], name='fixed-validation.json')
        run.log_artifact(artifact)
        url = run.url
    return dict(report=str(path), wandb_url=url, iteration=report['iteration'],
                manifest_sha256=report['manifest_sha256'])


def checkpoint_native(directory):
    from rwkv_conversion import (resolve_checkpoint, validate_model_config,
                                 convert_pipeline_stage, resolve_norm_offset, validate_rwkv_state_dict)

    folder, iteration = resolve_checkpoint(directory)
    files = sorted(Path(folder).glob('mp_rank_*/model_optim_rng.pt'))
    if not files:
        raise FileNotFoundError(f'No torch checkpoint shards under {folder}')
    metadata = torch.load(files[0], map_location='cpu', weights_only=False, mmap=True)['args']
    validate_model_config(metadata)
    tp, pp = metadata.tensor_model_parallel_size, metadata.pipeline_model_parallel_size
    if len(files) != tp * pp:
        raise ValueError('Missing or extra tensor/pipeline checkpoint shards')
    state = {}
    fields = ('num_layers', 'hidden_size', 'num_attention_heads', 'padded_vocab_size',
              'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'ffn_hidden_size',
              'D_AAA_LORA', 'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA',
              'num_layers_in_first_pipeline_stage', 'num_layers_in_last_pipeline_stage',
              'decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers')
    for stage in range(pp):
        shards = []
        for rank in range(tp):
            name = f'mp_rank_{rank:02d}' + (f'_{stage:03d}' if pp > 1 else '')
            data = torch.load(Path(folder) / name / 'model_optim_rng.pt', map_location='cpu',
                              weights_only=False, mmap=True)
            if data['iteration'] != iteration or any(getattr(data['args'], key, None) != getattr(metadata, key, None) for key in fields):
                raise ValueError('Checkpoint shard iteration/config mismatch')
            shards.append(data['model'])
        state.update(convert_pipeline_stage(shards, stage, metadata, resolve_norm_offset(metadata)))
    validate_rwkv_state_dict(state, metadata)
    return state, metadata, iteration, folder


def build_model(metadata):
    from megatron.training import global_vars
    from megatron.training.arguments import core_transformer_config_from_args
    from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
    from rwkv_megatron.rwkv7_model import RWKV7Model

    args = copy.deepcopy(metadata)
    for key in ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size', 'world_size'):
        setattr(args, key, 1)
    args.sequence_parallel = False
    args.rank = 0
    args.virtual_pipeline_model_parallel_size = None
    for key in ('num_layers_in_first_pipeline_stage', 'num_layers_in_last_pipeline_stage',
                'decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers'):
        setattr(args, key, None)
    args.gradient_accumulation_fusion = False
    args.recompute_granularity = None
    args.recompute_method = None
    args.recompute_num_layers = None
    global_vars._GLOBAL_ARGS = args
    config = core_transformer_config_from_args(args)
    for key in ('D_AAA_LORA', 'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA', 'rwkv_key_fusion', 'rwkv_fp8_scope'):
        setattr(config, key, getattr(args, key))
    config.perform_initialization = False
    config.use_cpu_initialization = False
    model = RWKV7Model(config, transformer_layer_spec=get_rwkv7_layer_spec(),
                      vocab_size=args.padded_vocab_size, max_sequence_length=args.max_position_embeddings,
                      parallel_output=False, share_embeddings_and_output_weights=False,
                      position_embedding_type='none').cuda().bfloat16().eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(model, manifest, tokenizer, max_new_tokens=32):
    records = []
    vocab = manifest['identity']['valid_vocab']
    for sample in manifest['rows']:
        tokens = torch.tensor([sample['tokens']], device='cuda')
        labels = torch.tensor(sample['labels'], device='cuda')
        mask = torch.tensor(sample['loss_mask'], device='cuda')
        positions = torch.arange(tokens.shape[1], device='cuda')[None]
        logits = model(tokens, positions, None)[0]
        parts = [score_logits(logits[i:i+64], labels[i:i+64], mask[i:i+64], vocab)
                 for i in range(0, len(labels), 64)]
        records.append({key: sum(part[key] for part in parts) for key in ('nll', 'tokens', 'correct')})
    generations = []
    for item in manifest['prompts']:
        prefix, generated = item['token_ids'], []
        if not prefix:
            raise ValueError('Empty fixed generation prompt')
        for _ in range(max_new_tokens):
            ids = prefix + generated
            length = ((len(ids) + 63) // 64) * 64
            tokens = torch.zeros((1, length), device='cuda', dtype=torch.long)
            tokens[0, :len(ids)] = torch.tensor(ids, device='cuda')
            logits = model(tokens, torch.arange(length, device='cuda')[None], None)
            token = int(logits[0, len(ids)-1, :vocab].argmax())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
        generations.append(dict(name=item['name'], prompt=item['prompt'], generated_ids=generated,
                                continuation=tokenizer.decode(generated, skip_special_tokens=False)))
    return dict(manifest_sha256=manifest['sha256'], metrics=aggregate(records), records=records,
                generations=generations, optimizer_steps=0, teacher_requests=0,
                decoding=dict(method='greedy', max_new_tokens=max_new_tokens,
                              input_padding='causal right padding to a multiple of 64'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rwkv-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    options = parser.parse_args()
    if options.max_new_tokens < 1 or options.samples < 1:
        parser.error('samples and max-new-tokens must be positive')
    if options.output.exists():
        parser.error('output already exists; use a new report path')
    settings = yaml.safe_load(options.rwkv_config.read_text())
    from transformers import AutoTokenizer
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    tokenizer = AutoTokenizer.from_pretrained(
        settings.get('rwkv_teacher_tokenizer') or settings['tokenizer_model'],
        trust_remote_code=True, local_files_only=True,
        use_fast=settings.get('rwkv_teacher_tokenizer_use_fast', True))
    manifest = fixed_manifest(options.manifest, settings, tokenizer, options.samples)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl')
    if dist.get_world_size() != 1:
        raise ValueError('Quality evaluator requires exactly one GPU/process')
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    started = time.monotonic()
    try:
        native, metadata, iteration, folder = checkpoint_native(options.checkpoint)
        if manifest['identity']['valid_vocab'] > metadata.padded_vocab_size:
            raise ValueError('Checkpoint vocabulary is smaller than the validation tokenizer')
        model = build_model(metadata)
        baseline = settings['load_pretrained_rwkv']
        initial = torch.load(baseline, map_location='cpu', weights_only=True, mmap=True)
        model.from_rwkv_lm(initial, False)
        del initial
        gc.collect()
        before = evaluate(model, manifest, tokenizer, options.max_new_tokens)
        print('INITIAL', json.dumps(before['metrics']), flush=True)
        model.from_rwkv_lm(native, False)
        del native
        gc.collect()
        after = evaluate(model, manifest, tokenizer, options.max_new_tokens)
        print('CHECKPOINT', iteration, json.dumps(after['metrics']), flush=True)
        report = dict(checkpoint=str(folder), iteration=iteration, initial_source=str(baseline),
                      manifest=str(options.manifest.resolve()), manifest_sha256=manifest['sha256'],
                      precision='production BF16 forward; FP32 token CE; FP64 NLL aggregation',
                      initial=before, checkpoint_result=after, comparison=compare(before, after),
                      seconds=time.monotonic()-started,
                      scope='Fixed validation and greedy continuations; not an instruction-following benchmark')
        write_json(options.output, report)
        print('QUALITY_REPORT', options.output, flush=True)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
