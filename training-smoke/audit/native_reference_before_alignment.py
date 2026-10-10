"""Differentiable RWKV7 reference using BlinkDL's documented slow torch formulas.

References pinned in reference-sources.json. No Megatron or fused TimeMix kernels.
The state has K,V orientation, transposed relative to BlinkDL's V,K RNN state.
"""
import math

import torch
import torch.nn.functional as F


class NativeReference(torch.nn.Module):
    def __init__(self,state,dtype,core='recurrent'):
        super().__init__()
        self.weights=torch.nn.ParameterDict({key.replace('.','__'):torch.nn.Parameter(value.cuda().to(dtype).clone())
                                             for key,value in state.items()})
        self.layers=max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))+1
        self.heads=state['blocks.0.att.r_k'].shape[0]
        self.head_size=state['blocks.0.att.r_k'].shape[1]
        self.core=core
        self.trace={}
        self.v_first=None

    def w(self,name):return self.weights[name.replace('.','__')]

    def mark(self,key,value):
        self.trace[key]=value.detach().cpu()
        return value

    def norm(self,x,prefix):
        return F.layer_norm(x,(x.shape[-1],),self.w(prefix+'.weight'),self.w(prefix+'.bias'),1e-5)

    def recurrent(self,r,w,k,v,a,b):
        dtype=r.dtype
        r,w,k,v,a,b=[x.float() for x in (r,w,k,v,a,b)]
        state=torch.zeros(r.shape[0],r.shape[2],r.shape[3],v.shape[3],device=r.device)
        out=[]
        for t in range(r.shape[1]):
            sa=(state*a[:,t].unsqueeze(-1)).sum(-2)
            state=state*w[:,t].exp().unsqueeze(-1)+k[:,t].unsqueeze(-1)*v[:,t].unsqueeze(-2)
            state=state+b[:,t].unsqueeze(-1)*sa.unsqueeze(-2)
            out.append((state*r[:,t].unsqueeze(-1)).sum(-2))
        return torch.stack(out,dim=1).to(dtype)

    def forward(self,tokens):
        x=F.embedding(tokens,self.w('emb.weight'))
        x=self.mark('ln0',self.norm(x,'blocks.0.ln0'))
        first=None
        for layer in range(self.layers):
            block=f'blocks.{layer}.';att=block+'att.'
            n=self.mark(f'{layer}.norm1',self.norm(x,block+'ln1'))
            shift=F.pad(n[:,:-1],(0,0,1,0))-n
            mix={ch:self.mark(f'{layer}.mix-{ch}',n+shift*self.w(att+'x_'+ch)) for ch in 'rwkvag'}
            r=self.mark(f'{layer}.r',F.linear(mix['r'],self.w(att+'receptance.weight')))
            k=self.mark(f'{layer}.k-raw',F.linear(mix['k'],self.w(att+'key.weight')))
            v=self.mark(f'{layer}.v-raw',F.linear(mix['v'],self.w(att+'value.weight')))
            decay=self.w(att+'w0')+torch.tanh(mix['w']@self.w(att+'w1'))@self.w(att+'w2')
            # exp(-softplus(-decay)-0.5) = exp(-0.5)*sigmoid(decay).
            w=self.mark(f'{layer}.log-decay',-math.exp(-0.5)*torch.sigmoid(decay))
            a=torch.sigmoid(self.w(att+'a0')+(mix['a']@self.w(att+'a1'))@self.w(att+'a2'))
            g=torch.sigmoid(mix['g']@self.w(att+'g1'))@self.w(att+'g2')
            if layer==0:
                first=v;first.retain_grad();self.v_first=first
            else:
                gate=torch.sigmoid(self.w(att+'v0')+(mix['v']@self.w(att+'v1'))@self.w(att+'v2'))
                v=v+(first-v)*gate
            v=self.mark(f'{layer}.v',v)
            kk=F.normalize((k*self.w(att+'k_k')).reshape(*k.shape[:2],self.heads,self.head_size),dim=-1)
            k=self.mark(f'{layer}.k',k*(1+(a-1)*self.w(att+'k_a')))
            reshape=lambda t:t.reshape(*t.shape[:2],self.heads,self.head_size).contiguous()
            r,w,k,v=map(reshape,(r,w,k,v))
            a=reshape(a)
            if self.core=='fla':
                from fla.ops.rwkv7 import chunk_rwkv7
                o=chunk_rwkv7(r,w,k,v,-kk,kk*a,scale=1.,safe_gate=True,chunk_size=64)[0]
            else:o=self.recurrent(r,w,k,v,-kk,kk*a)
            o=self.mark(f'{layer}.wkv',o.flatten(2))
            o=F.group_norm(o.reshape(-1,o.shape[-1]),self.heads,self.w(att+'ln_x.weight'),
                           self.w(att+'ln_x.bias'),self.head_size*1e-5).reshape_as(o)
            correction=((r*k*self.w(att+'r_k')).sum(-1,keepdim=True)*v).flatten(2)
            o=self.mark(f'{layer}.att',F.linear((o+correction)*g,self.w(att+'output.weight')))
            x=x+o
            n=self.mark(f'{layer}.norm2',self.norm(x,block+'ln2'))
            shift=F.pad(n[:,:-1],(0,0,1,0))-n
            mixed=n+shift*self.w(block+'ffn.x_k')
            o=F.linear(torch.relu(F.linear(mixed,self.w(block+'ffn.key.weight'))).square(),
                       self.w(block+'ffn.value.weight'))
            self.mark(f'{layer}.mlp',o)
            x=self.mark(f'{layer}.hidden',x+o)
        x=self.mark('final-norm',self.norm(x,'ln_out'))
        return self.mark('logits',F.linear(x,self.w('head.weight')))

    def native_gradients(self,selected=None):
        result={}
        for name,param in self.weights.items():
            key=name.replace('__','.')
            if key.startswith('blocks.0.att.v') and key.rsplit('.',1)[-1] in ('v0','v1','v2'):continue
            if selected is not None and key not in selected:continue
            result[key]=param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param).cpu()
        return result
