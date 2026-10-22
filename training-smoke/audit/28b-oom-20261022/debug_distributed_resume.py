import copy
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '3rdparty/Megatron-LM-0.11.0'))
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
parallel_state.initialize_model_parallel(context_parallel_size=2)
pg = ProcessGroupCollection.use_mpu_process_groups()
config = TransformerConfig(num_layers=1, hidden_size=256, num_attention_heads=4,
                           context_parallel_size=2, bf16=True, params_dtype=torch.bfloat16)


def build(state=None):
    module = torch.nn.Module()
    module.config = config
    module.weights = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(256, 256, dtype=torch.bfloat16, device='cuda'))
        for _ in range(8)])
    if state is not None:
        module.load_state_dict(state)
    model = DistributedDataParallel(config, DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True), module, pg_collection=pg)
    opt = get_megatron_optimizer(OptimizerConfig(
        bf16=True, optimizer='adam', lr=1e-6, use_distributed_optimizer=True,
        optimizer_cpu_offload=True, optimizer_offload_fraction=0.5,
        use_precision_aware_optimizer=True, clip_grad=0.0), [model])
    return model, opt


def step(model, opt):
    model.zero_grad_buffer()
    opt.zero_grad()
    for p in model.parameters():
        p.main_grad.fill_(0.125)
    assert opt.step()[0]
    torch.cuda.synchronize()


a, oa = build()
for _ in range(3):
    step(a, oa)
model_state = copy.deepcopy(a.module.state_dict())
meta = copy.deepcopy(oa.state_dict())
ckpt = ROOT / 'training-smoke/audit/28b-oom-20261022/toy-distrib-optim.pt'
oa.save_parameter_state(str(ckpt))
b, ob = build(model_state)
ob.load_state_dict(meta)
ob.load_parameter_state(str(ckpt))
for name, value in b.module.state_dict().items():
    torch.testing.assert_close(value, model_state[name], rtol=0, atol=0)
for part in ob.chained_optimizers:
    for sub in part.optimizer.sub_optimizers:
        for param, state in sub.state.items():
            for key in ['exp_avg', 'exp_avg_sq']:
                assert state[key].device == param.device, (key, state[key].device, param.device)
                assert state[key].shape == param.shape
def dump(opt):
    result=[]
    for part in opt.chained_optimizers:
        for sub in part.optimizer.sub_optimizers:
            for group in sub.param_groups:
                for param in group['params']:
                    state=sub.state[param]
                    result.append((type(sub).__name__, str(param.device), param.flatten()[0].item(),
                        group.get('step'), {key: val.flatten()[0].item() for key,val in state.items() if isinstance(val,torch.Tensor)}))
    return result
print('rank',rank,'before',dump(oa),dump(ob),flush=True)
for loop in range(2):
    step(a, oa)
    step(b, ob)
    print('rank',rank,'after',loop,dump(oa),dump(ob),flush=True)
    for name, value in b.module.state_dict().items():
        torch.testing.assert_close(value, a.module.state_dict()[name], rtol=0, atol=0)
print('rank', rank, 'distributed continuation exact', flush=True)
dist.destroy_process_group()
