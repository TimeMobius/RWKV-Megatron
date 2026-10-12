"""Frozen pre-change payload for controlled comparisons; never production imports."""
import ast
from pathlib import Path

import rwkv_megatron.rwkv7_layer_specs as specs
from megatron.core.pipeline_parallel import schedules

PATH=Path(__file__).resolve().parent/'before'


def install():
    tree=ast.parse((PATH/'rwkv7_layer_specs.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RWKV7TimeMix')
    forward=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    module=ast.Module(body=[forward],type_ignores=[]);ast.fix_missing_locations(module)
    scope=dict(specs.__dict__)
    exec(compile(module,str(PATH/'rwkv7_layer_specs.py'),'exec'),scope)
    specs.RWKV7TimeMix.forward=scope['forward']
    tree=ast.parse((PATH/'schedules.py').read_text())
    forward=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='get_tensor_shapes')
    module=ast.Module(body=[forward],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,str(PATH/'schedules.py'),'exec'),schedules.__dict__)
