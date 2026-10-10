"""Same real 3B inputs, current/official operators and FP32 formula gradients."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import einsum

from blinkdl_reference import fast_namespace
from check import metric
from rwkv_megatron.rwkv7_layer_specs import mix_inputs, update_key

PATH=Path(__file__).resolve().parent


def key_current(k,a,beta,alpha):
    kk=F.normalize((k*beta).reshape(*k.shape[:2],-1,64),dim=-1).flatten(2)
    return update_key(k,a,alpha),-kk,kk*a


def key_formula(k,a,beta,alpha):
    kk=F.normalize((k*beta).reshape(*k.shape[:2],-1,64),dim=-1).flatten(2)
    return k*(1+(a-1)*alpha),-kk,kk*a


def vres_current(v,first,bias,pre):
    return torch.lerp(v,first,torch.sigmoid(pre+bias))


def vres_formula(v,first,bias,pre):
    return v+(first-v)*torch.sigmoid(pre+bias)


def norm_correction(x,r,k,v,rk,weight,bias,g):
    heads=x.shape[-1]//64
    normalized=F.group_norm(x.reshape(-1,x.shape[-1]),heads,weight,bias,64e-5).reshape_as(x)
    reshape=lambda z:z.reshape(*z.shape[:2],heads,64)
    correction=((reshape(r)*reshape(k)*rk).sum(-1,keepdim=True)*reshape(v)).flatten(2)
    return (normalized+correction)*g


def norm_formula(x,r,k,v,rk,weight,bias,g):
    reshape=lambda z:z.reshape(*z.shape[:2],-1,64)
    grouped=reshape(x)
    centered=grouped-grouped.mean(-1,keepdim=True)
    normalized=(centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+64e-5)).flatten(2)
    correction=((reshape(r)*reshape(k)*rk).sum(-1,keepdim=True)*reshape(v)).flatten(2)
    return (normalized*weight+bias+correction)*g


def ffn_current(x,coeff,up,down):
    delta=F.pad(x[:,:-1],(0,0,1,0))-x
    mixed=torch.addcmul(x,delta,coeff)
    return F.linear(F.relu(F.linear(mixed,up)).square(),down)


def ffn_formula(x,coeff,up,down):
    delta=F.pad(x[:,:-1],(0,0,1,0))-x
    return F.linear(F.relu(F.linear(x+delta*coeff,up)).square(),down)


def measure(inputs,current,official,formula,names):
    oracle=[value.detach().float().requires_grad_() for value in inputs]
    ref=formula(*oracle)
    ref=(ref,) if isinstance(ref,torch.Tensor) else ref
    dy=[torch.randn_like(value).to(torch.bfloat16) for value in ref]
    ref_grad=torch.autograd.grad(ref,oracle,[d.float() for d in dy])
    result={}
    current_outputs=None
    for label,fn,dtype in [('current_bf16',current,torch.bfloat16),('official_bf16',official,torch.bfloat16),('current_fp32',current,torch.float32)]:
        tensors=[value.detach().to(dtype).requires_grad_() for value in inputs]
        out=fn(*tensors)
        out=(out,) if isinstance(out,torch.Tensor) else out
        grads=torch.autograd.grad(out,tensors,[d.to(dtype) for d in dy])
        checks={'output':{str(i):metric(value,expected) for i,(value,expected) in enumerate(zip(out,ref))},
                'gradient':{name:metric(actual,expected) for name,actual,expected in zip(names,grads,ref_grad)}}
        checks['aggregate_gradient_relative_l2']=(sum((v['relative_l2']*v['reference_l2'])**2 for v in checks['gradient'].values())/
                                                 sum(v['reference_l2']**2 for v in checks['gradient'].values()))**0.5
        result[label]=checks
        if label=='current_bf16':current_outputs=tuple(t.detach().clone() for t in out)
        if label=='official_bf16':
            result['current_vs_official_forward']={'exact':all(torch.equal(a,b) for a,b in zip(current_outputs,out)),
                'output':{str(i):metric(a,b) for i,(a,b) in enumerate(zip(current_outputs,out))}}
        if dtype==torch.float32:
            # Formula check; BF16 errors are measured, not silently accepted.
            assert max(m['relative_l2'] for m in checks['gradient'].values())<1e-3
            assert max(m['relative_l2'] for m in checks['output'].values())<1e-5
        for group in ('gradient','output'):
            assert all(m['finite'] for m in checks[group].values())
    return result


def main():
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False
    ns,_=fast_namespace()
    # Confirm the loader can safely be reused without duplicate registrations.
    ns2,_=fast_namespace()
    assert ns is not ns2 and ns['RWKV7_CLAMPW_CUDA'] is ns2['RWKV7_CLAMPW_CUDA']
    state=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth',map_location='cpu',weights_only=True)
    capture=torch.load(PATH/'full-new-replay-core.pth',map_location='cpu',weights_only=True)
    trace=capture['trace']
    rows={}
    def w(name):return state[name].cuda().to(torch.bfloat16)
    def t(layer,name):return trace[f'{layer}.{name}'].cuda().to(torch.bfloat16)
    for layer in (0,15,31):
        prefix=f'blocks.{layer}.att.'
        n=t(layer,'norm1').transpose(0,1)
        delta=torch.cat((torch.zeros_like(n[:1]),n[:-1]),dim=0)-n
        mixed=dict(zip('rwkvag',mix_inputs(n,delta,[w(prefix+'x_'+ch).flatten() for ch in 'rwkvag'])))
        def lora(ch,family,activation=lambda x:x):
            neck=einsum(mixed[ch],w(prefix+family+'1'),'t b d, d r -> t b r')
            return einsum(activation(neck),w(prefix+family+'2'),'t b r, r d -> t b d').transpose(0,1)
        learn=torch.sigmoid(lora('a','a')+w(prefix+'a0'))
        gate=lora('g','g',torch.sigmoid)
        key_inputs=[t(layer,'k-raw'),learn,w(prefix+'k_k').flatten(),w(prefix+'k_a').flatten()]
        # Verify reconstructed operator inputs reproduce saved production values.
        key_out=key_current(*key_inputs)
        for value,name in zip(key_out,('core-k','core-a','core-b')):
            assert torch.equal(value,t(layer,name)),(layer,name,'reconstructed input mismatch')
        official_key=lambda k,a,beta,alpha:ns['tmix_kk_pre_bf16_v5'](k,beta,a,alpha)
        rows[f'{layer}.key']=measure(key_inputs,key_current,official_key,key_formula,('k','learn','beta','alpha'))
        norm_inputs=[t(layer,name) for name in ('wkv','core-r','core-k','core-v')]+[w(prefix+'r_k'),w(prefix+'ln_x.weight'),w(prefix+'ln_x.bias'),gate]
        assert torch.equal(norm_correction(*norm_inputs),t(layer,'project-input')),(layer,'norm reconstruction mismatch')
        rows[f'{layer}.norm_correction_gate']=measure(norm_inputs,norm_correction,ns['tmix_lnx_rkvres_xg_bf16_v1'],norm_formula,('x','r','k','v','rho','weight','bias','gate'))
        if layer:
            v_inputs=[t(layer,'v-raw'),t(0,'v-raw'),w(prefix+'v0').flatten(),lora('v','v')]
            assert torch.equal(vres_current(*v_inputs),t(layer,'core-v')),(layer,'v reconstructed input mismatch')
            rows[f'{layer}.value_residual']=measure(v_inputs,vres_current,ns['tmix_vres_gate_bf16_v3'],vres_formula,('v','first','bias','pre'))
        ffn_prefix=f'blocks.{layer}.ffn.'
        ffn_inputs=[t(layer,'norm2'),w(ffn_prefix+'x_k'),w(ffn_prefix+'key.weight'),w(ffn_prefix+'value.weight')]
        assert torch.equal(ffn_current(*ffn_inputs),t(layer,'mlp')),(layer,'ffn reconstruction mismatch')
        official_ffn=lambda x,c,up,down:ns['_CmixLayerV2Fn'].apply(x,c.flatten(),up,down)
        rows[f'{layer}.ffn']=measure(ffn_inputs,ffn_current,official_ffn,ffn_formula,('x','coeff','up','down'))
        print('OPERATOR DONE',layer,flush=True)
    (PATH/'operator-precision-results.json').write_text(json.dumps({'input':'fixed 3B MiniPile seq128; identical input and upstream gradient per operator',
        'layers':[0,15,31],'bf16_accuracy_certified':False,'fp32_formula_checks_passed':True,'results':rows},indent=2)+'\n')


if __name__=='__main__':main()
