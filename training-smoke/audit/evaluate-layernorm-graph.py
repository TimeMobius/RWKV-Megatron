"""Reuse captured inputs to distinguish eager dispatch from CUDA Graph replay."""
import json
import statistics
import torch
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('norm_evaluation', PATH / 'evaluate-layernorm.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def graph_bench(action):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            action()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        action()
    for _ in range(10):
        graph.replay()
    trials = [audit.elapsed(graph.replay, 100) for _ in range(5)]
    return {'median': statistics.median(trials), 'trials': trials}


def main():
    torch.cuda.set_device(0)
    audit.audit.FULL = True
    audit.audit.DIMS = {'layers': 32, 'hidden': 2560, 'heads': 40, 'ffn': 10240,
                       'vocab': 65536, 'sequence': 128, 'batch': 1, 'microbatches': 1}
    cfg = audit.audit.config_for(1, 1, False, False, fp32=False)
    captures = torch.load(PATH / 'layernorm-inputs-current.pth', weights_only=True)
    data = captures['ln0']
    result = json.loads((PATH / 'layernorm-evaluation.json').read_text())
    benchmarks = []
    for rows, batch in ((128, 1), (512, 1), (4096, 1), (4096, 4)):
        base = data['input'].reshape(-1, 2560)
        x0 = base.repeat((rows * batch + 127) // 128, 1)[:rows * batch].reshape(rows, batch, 2560).cuda()
        dy = torch.randn_like(x0)
        values = {}
        for backend in ('torch', 'apex', 'te'):
            module = audit.build(backend, cfg, data['weight'].cuda(), data['bias'].cuda())
            x = x0.clone().requires_grad_()
            def forward():
                with torch.no_grad():
                    module(x)
            def train():
                module.zero_grad(set_to_none=True)
                x.grad = None
                module(x).backward(dy)
            values[backend] = {}
            for mode, action in (('forward_us', forward), ('forward_backward_us', train)):
                try:
                    values[backend][mode] = graph_bench(action)
                except Exception as exc:
                    values[backend][mode] = {'error': repr(exc)}
        benchmarks.append({'shape': [rows, batch, 2560], 'results': values})
        print('GRAPH', [rows, batch, 2560],
              {b: {k: round(v['median'], 2) if 'median' in v else v for k, v in d.items()}
               for b, d in values.items()}, flush=True)
    result['graph_benchmarks'] = benchmarks
    result['graph_benchmark'] = 'Same repeated BF16 ln0 inputs; CUDA Graph replay,5 trials of100 iterations;10 warmup. Operator capture only, not full-model graphs.'
    (PATH / 'layernorm-evaluation.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
