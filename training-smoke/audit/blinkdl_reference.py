"""Execute pinned BlinkDL forward bodies instead of reimplementing their graph.

Only training/demo side effects and MyFunction/JIT decorators are excluded.
FP32 uses the official simplified TimeMix/FFN bodies with an independent torch
recurrence; BF16 uses the main training bodies and their original CUDA ops.
Plain CE is supplied by the caller; official L2Wrap is intentionally separate.
"""
import ast
import copy
import hashlib
import os
import subprocess
from pathlib import Path
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from native_reference import NativeReference

PATH = Path(__file__).resolve().parent
SOURCE = PATH.parent / 'reference-repos/RWKV-LM/RWKV-v7/train_temp'
COMMIT = '20be0f8cf0bdf506d8c0030c87a9a3416f95b143'


def source_method(path, class_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'))
    method.decorator_list = []
    # The executable body is copied directly, with no algebra/layout edits.
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    assert ast.dump(ast.Module(body=method.body, type_ignores=[])) == ast.dump(ast.Module(body=original.body, type_ignores=[]))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace.pop('forward')


def fast_namespace():
    from torch.utils.cpp_extension import load

    path = SOURCE / 'src/model.py'
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ['RWKV_MY_TESTING'] = 'x070'
    # Select the documented baseline clampw variant, not the optional H100 v3.
    os.environ['RWKV_KERNEL'] = ''
    os.environ['RWKV_HEAD_SIZE'] = '64'
    os.environ.setdefault('MAX_JOBS', '2')
    loaded = []

    def isolated_load(**kwargs):
        sources = [SOURCE / name for name in kwargs['sources']]
        kwargs['sources'] = [str(p) for p in sources]
        build = PATH.parent / 'cache/torch-extensions-blinkdl' / kwargs['name']
        build.mkdir(parents=True, exist_ok=True)
        kwargs['build_directory'] = str(build)
        # Conda splits CUDA headers out of CUDA_HOME/include. Supply the actual
        # installed header location without touching official source or flags.
        kwargs['extra_include_paths'] = [str(Path(os.environ['CUDA_HOME']) / 'targets/x86_64-linux/include')]
        kwargs['extra_include_paths'].extend(str(p) for p in sorted((Path(torch.__file__).parent.parent / 'nvidia').glob('*/include')))
        loaded.extend(sources)
        return load(**kwargs)

    namespace = dict(torch=torch, nn=nn, F=F, os=os, load=isolated_load,
                     HEAD_SIZE=64, __name__=__name__)
    tree = ast.parse(path.read_text())
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.If) and "'x070'" in ast.unparse(node.test))
    end = next(i + 1 for i, node in enumerate(tree.body)
               if isinstance(node, ast.If) and any(isinstance(child, ast.FunctionDef)
               and child.name == 'tmix_vres_gate_bf16_v3' for child in node.body))
    # Stop before loss/head extension loads; they do not participate in features.
    nodes = [copy.deepcopy(node) for node in tree.body[start:end]]
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('_tmix_'):
            # Execute the exact wrapper body eagerly. Its CUDA call and autograd
            # registration stay official; no ScriptModule/trainer is imported.
            node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, loaded


class BlinkDLReference(NativeReference):
    def __init__(self, state, dtype):
        super().__init__(state, dtype)
        checkout = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        assert checkout == COMMIT, checkout
        self.dtype = dtype
        self.source_files = []
        if dtype == torch.bfloat16:
            namespace, loaded = fast_namespace()
            path = SOURCE / 'src/model.py'
            tmix_name, ffn_name = 'RWKV_Tmix_x070', 'RWKV_CMix_x070'
            core_name = 'RWKV7_CLAMPW_CUDA'
            raw_core = namespace[core_name]
            self.source_files.extend(loaded)
        else:
            namespace = dict(torch=torch, nn=nn, F=F)
            path = SOURCE / 'rwkv7_train_simplified.py'
            tmix_name, ffn_name = 'RWKV_Tmix_x070', 'FFN'
            core_name = 'RUN_CUDA_RWKV7g'
            raw_core = self.torch_core
        self.source_files.append(path)
        tmix_forward = source_method(path, tmix_name, namespace)
        ffn_forward = source_method(path, ffn_name, namespace)
        # Official Block has ln0 and preserves the v_first object unchanged.
        block_path = SOURCE / 'src/model.py'
        block_forward = source_method(block_path, 'Block', namespace)
        self.source_files.append(block_path)
        self._current_layer = 0

        def traced_core(r, w, k, v, a, b):
            index = self._current_layer
            for name, value in zip(('r', 'k', 'v', 'a', 'b'), (r, k, v, a, b)):
                self.mark(f'{index}.core-{name}', value)
            # Positive decay magnitude, matching Megatron's public core input.
            decay = (torch.exp(torch.tensor(-0.5)) * torch.sigmoid(w)
                     if dtype == torch.bfloat16 else w.exp())
            self.mark(f'{index}.core-decay', decay)
            return self.mark(f'{index}.wkv', raw_core(r, w, k, v, a, b))

        namespace[core_name] = traced_core
        self.blocks = nn.ModuleList()
        for index in range(self.layers):
            prefix = f'blocks.{index}.'
            att = nn.Module()
            att.layer_id, att.n_head = index, self.heads
            att.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            for name in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g', 'w0', 'w1', 'w2',
                         'a0', 'a1', 'a2', 'v0', 'v1', 'v2', 'g1', 'g2', 'k_k', 'k_a', 'r_k'):
                key = prefix + 'att.' + name
                if key.replace('.', '__') in self.weights:
                    setattr(att, name, self.w(key))
            for name in ('receptance', 'key', 'value', 'output'):
                setattr(att, name, self.linear(prefix + 'att.' + name + '.weight'))
            att.ln_x = nn.GroupNorm(self.heads, self.heads * self.head_size, eps=64e-5, device='meta')
            att.ln_x.weight = self.w(prefix + 'att.ln_x.weight')
            att.ln_x.bias = self.w(prefix + 'att.ln_x.bias')
            att.forward = MethodType(tmix_forward, att)
            ffn = nn.Module()
            ffn.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            ffn.x_k = self.w(prefix + 'ffn.x_k')
            ffn.key = self.linear(prefix + 'ffn.key.weight')
            ffn.value = self.linear(prefix + 'ffn.value.weight')
            ffn.forward = MethodType(ffn_forward, ffn)
            block = nn.Module()
            block.layer_id = index
            block.att, block.ffn = att, ffn
            for name in ('ln0', 'ln1', 'ln2'):
                if name != 'ln0' or index == 0:
                    setattr(block, name, self.layer_norm(prefix + name))
            block.forward = MethodType(block_forward, block)
            self.blocks.append(block)
            if index == 0:
                block.ln0.register_forward_hook(self.hook('ln0'))
            for key, module in (('norm1', block.ln1), ('norm2', block.ln2),
                                ('r', att.receptance), ('k-raw', att.key), ('v-raw', att.value),
                                ('att', att), ('mlp', ffn), ('hidden', block)):
                module.register_forward_hook(self.hook(f'{index}.{key}'))
        self.output_norm = self.layer_norm('ln_out')
        self.output_norm.register_forward_hook(self.hook('final-norm'))

    def linear(self, key):
        weight = self.w(key)
        module = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='meta')
        module.weight = weight
        return module

    def layer_norm(self, prefix):
        module = nn.LayerNorm(self.w(prefix + '.weight').numel(), eps=1e-5, device='meta')
        module.weight, module.bias = self.w(prefix + '.weight'), self.w(prefix + '.bias')
        return module

    def hook(self, key):
        def record(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            self.mark(key, value)
            if key == '0.v-raw':
                self.v_first = value
        return record

    def torch_core(self, r, w, k, v, a, b):
        # Independent FP32 recurrence in K,V orientation. Reshape only inside
        # this core; official correction creates its own separate v.view.
        reshape = lambda value: value.view(*value.shape[:2], self.heads, self.head_size)
        r, w, k, v, a, b = map(reshape, (r, w, k, v, a, b))
        return self.recurrent(r, -w.exp(), k, v, a, b).flatten(2)

    def forward(self, tokens):
        x = F.embedding(tokens, self.w('emb.weight'))
        first = torch.empty_like(x)
        for index, block in enumerate(self.blocks):
            self._current_layer = index
            x, first = block(x, first)
            assert first is self.v_first, 'official v_first identity changed'
        x = self.output_norm(x)
        logits = F.linear(x, self.w('head.weight'))
        return self.mark('logits', logits)

    def provenance(self):
        return dict(repository='BlinkDL/RWKV-LM', commit=COMMIT,
                    variant='official main BF16 CUDA' if self.dtype == torch.bfloat16 else 'official simplified FP32 body + independent torch recurrence',
                    exact_source_forward_bodies=True, v_first_transport='direct object reuse, no carrier',
                    objective='plain CE; no official L2Wrap', jit='forward/wrapper bodies executed eagerly',
                    cuda_kernel_variant='baseline clampw' if self.dtype == torch.bfloat16 else None,
                    source_sha256={str(path.relative_to(SOURCE)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(set(self.source_files))})
