"""Differentiable RWKV7 reference using BlinkDL's documented slow torch formulas.

References pinned in reference-sources.json. Defaults to independent torch formulas.
Diagnostic options can share precision primitives to isolate model wiring.
The state has K,V orientation, transposed relative to BlinkDL's V,K RNN state.
"""
import math

import torch
import torch.nn.functional as F


class NativeReference(torch.nn.Module):
    def __init__(self,state,dtype,core='recurrent',aligned=False,apex_norms=False,shared_pointops=False,te_final=False,carrier_chain=False,te_norms=False,split_value_views=False):
        super().__init__()
        self.weights=torch.nn.ParameterDict({key.replace('.','__'):torch.nn.Parameter(value.cuda().to(dtype).clone())
                                             for key,value in state.items()})
        self.layers=max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))+1
        self.heads=state['blocks.0.att.r_k'].shape[0]
        self.head_size=state['blocks.0.att.r_k'].shape[1]
        self.core=core
        self.aligned=aligned
        self.apex_norms=apex_norms
        self.shared_pointops=shared_pointops
        self.te_final=te_final
        self.carrier_chain=carrier_chain
        self.te_norms=te_norms
        self.split_value_views=split_value_views
        self.layer_norms=torch.nn.ModuleDict()
        if te_norms:
            from transformer_engine.pytorch import LayerNorm
            prefixes=['blocks.0.ln0']+[f'blocks.{i}.{norm}' for i in range(self.layers) for norm in ('ln1','ln2')]
            for prefix in prefixes:
                module=LayerNorm(state[prefix+'.weight'].numel(),eps=1e-5,params_dtype=dtype,device='cuda')
                module.weight=self.w(prefix+'.weight');module.bias=self.w(prefix+'.bias')
                self.layer_norms[prefix.replace('.','__')]=module
        if te_final:
            from transformer_engine.pytorch import LayerNorm
            self.final_norm=LayerNorm(state['ln_out.weight'].numel(),eps=1e-5,params_dtype=dtype,device='cuda')
            self.final_norm.weight=self.w('ln_out.weight')
            self.final_norm.bias=self.w('ln_out.bias')
        self.trace={}
        self.live={}
        self.v_first=None

    def w(self,name):return self.weights[name.replace('.','__')]

    def mark(self,key,value):
        self.trace[key]=(value.transpose(0,1) if self.aligned else value).detach().cpu()
        if value.requires_grad:
            value.retain_grad();self.live[key]=value
        return value

    def norm(self,x,prefix):
        if prefix=='ln_out' and self.te_final:return self.final_norm(x)
        if self.te_norms and prefix!='ln_out':return self.layer_norms[prefix.replace('.','__')](x)
        if self.apex_norms:
            from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
            return FusedLayerNormAffineFunction.apply(x,self.w(prefix+'.weight'),self.w(prefix+'.bias'),(x.shape[-1],),1e-5,False)
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
        if self.aligned:x=x.transpose(0,1).contiguous()
        x=self.mark('ln0',self.norm(x,'blocks.0.ln0'))
        first=None
        for layer in range(self.layers):
            block=f'blocks.{layer}.';att=block+'att.'
            n=self.mark(f'{layer}.norm1',self.norm(x,block+'ln1'))
            if self.carrier_chain and first is not None:
                normalized_carrier=torch.cat((n.unsqueeze(0),first.unsqueeze(0)),dim=0)
                n,first=normalized_carrier[0],normalized_carrier[1]
            shift=(torch.cat((torch.zeros_like(n[:1]),n[:-1]),dim=0)-n) if self.aligned else F.pad(n[:,:-1],(0,0,1,0))-n
            mix={ch:self.mark(f'{layer}.mix-{ch}',
                torch.addcmul(n.float(),shift.float(),self.w(att+'x_'+ch).float()).to(n.dtype) if self.aligned
                else n+shift*self.w(att+'x_'+ch)) for ch in 'rwkvag'}
            if self.shared_pointops:
                from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
                outputs=fused_addcmul_rwkv7(n.reshape(1,-1,n.shape[-1]),shift.reshape(1,-1,n.shape[-1]),
                                          *[self.w(att+'x_'+ch).reshape(-1) for ch in 'rwkvag'])
                mix={ch:value.reshape_as(n) for ch,value in zip('rwkvag',outputs)}
            r=self.mark(f'{layer}.r',F.linear(mix['r'],self.w(att+'receptance.weight')))
            k=self.mark(f'{layer}.k-raw',F.linear(mix['k'],self.w(att+'key.weight')))
            raw_key=k
            v=self.mark(f'{layer}.v-raw',F.linear(mix['v'],self.w(att+'value.weight')))
            def lora(z,family,activation=lambda x:x):
                down=self.w(att+family+'1');up=self.w(att+family+'2')
                if self.aligned:return torch.einsum('tbr,rd->tbd',activation(torch.einsum('tbd,dr->tbr',z,down)),up)
                return activation(z@down)@up
            decay=lora(mix['w'],'w',torch.tanh)+self.w(att+'w0')
            # exp(-softplus(-decay)-0.5) = exp(-0.5)*sigmoid(decay).
            w=self.mark(f'{layer}.log-decay',-(torch.exp(torch.tensor(-0.5))*torch.sigmoid(decay) if self.aligned else math.exp(-0.5)*torch.sigmoid(decay)))
            a=torch.sigmoid(lora(mix['a'],'a')+self.w(att+'a0'))
            g=lora(mix['g'],'g',torch.sigmoid)
            if layer==0:
                first=v;first.retain_grad();self.v_first=first
                if self.carrier_chain:
                    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
                    first=gather_from_tensor_model_parallel_region(v)
            else:
                gate=torch.sigmoid(lora(mix['v'],'v')+self.w(att+'v0'))
                v=torch.lerp(v,first,gate) if self.aligned else v+(first-v)*gate
            v=self.mark(f'{layer}.v',v)
            correction_value=v
            kk=F.normalize((k*self.w(att+'k_k')).reshape(*k.shape[:2],self.heads,self.head_size),dim=-1)
            k=self.mark(f'{layer}.k', (k.float()*torch.addcmul(torch.ones_like(k,dtype=torch.float32),a.float()-1,self.w(att+'k_a').float())).to(k.dtype) if self.aligned else k*(1+(a-1)*self.w(att+'k_a')))
            if self.shared_pointops:
                from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
                raw=raw_key
                k=fused_k_rwkv7(raw.reshape(1,-1,raw.shape[-1]),a.reshape(1,-1,a.shape[-1]),
                               self.w(att+'k_a').reshape(-1)).reshape_as(raw)
            reshape=lambda t:t.reshape(*t.shape[:2],self.heads,self.head_size).contiguous()
            r,w,k,v=map(reshape,(r,w,k,v))
            a=reshape(a)
            if self.aligned:r,w,k,v,a,kk=[t.transpose(0,1).contiguous() for t in (r,w,k,v,a,kk)]
            for name,value in zip(('r','k','v','decay','a','b'),(r,k,v,-w,-kk,kk*a)):
                shaped=value.flatten(2).transpose(0,1) if self.aligned else value.flatten(2)
                self.mark(f'{layer}.core-{name}',shaped)
            if self.core=='fla':
                from fla.ops.rwkv7 import chunk_rwkv7
                o=chunk_rwkv7(r,w,k,v,-kk,kk*a,scale=1.,safe_gate=True,chunk_size=64)[0]
            else:o=self.recurrent(r,w,k,v,-kk,kk*a)
            if self.aligned:o=o.transpose(0,1).contiguous()
            o=self.mark(f'{layer}.wkv',o.flatten(2))
            o=F.group_norm(o.reshape(-1,o.shape[-1]),self.heads,self.w(att+'ln_x.weight'),
                           self.w(att+'ln_x.bias'),self.head_size*1e-5).reshape_as(o)
            self.mark(f'{layer}.groupnorm',o)
            if self.split_value_views:
                # Diagnostic only: production independently views value for
                # WKV and output correction. A shared view changes the BF16
                # addition grouping when layer-zero also feeds the carrier.
                correction_value=reshape(correction_value)
                if self.aligned:correction_value=correction_value.transpose(0,1)
            else:correction_value=v
            correction=((r*k*self.w(att+'r_k')).sum(-1,keepdim=True)*correction_value).flatten(2)
            if self.aligned:correction=correction.transpose(0,1)
            project_input=self.mark(f'{layer}.project-input',(o+correction)*g)
            o=self.mark(f'{layer}.att',F.linear(project_input,self.w(att+'output.weight')))
            if self.carrier_chain:
                att_carrier=torch.stack((o,first))
                o,first=att_carrier[0],att_carrier[1]
            x=x+o
            n=self.mark(f'{layer}.norm2',self.norm(x,block+'ln2'))
            shift=(torch.cat((torch.zeros_like(n[:1]),n[:-1]),dim=0)-n) if self.aligned else F.pad(n[:,:-1],(0,0,1,0))-n
            mixed=torch.addcmul(n,shift,self.w(block+'ffn.x_k')) if self.aligned else n+shift*self.w(block+'ffn.x_k')
            o=F.linear(torch.relu(F.linear(mixed,self.w(block+'ffn.key.weight'))).square(),
                       self.w(block+'ffn.value.weight'))
            self.mark(f'{layer}.mlp',o)
            x=self.mark(f'{layer}.hidden',x+o)
            if self.carrier_chain:
                layer_carrier=torch.cat((x.unsqueeze(0),first.unsqueeze(0)),dim=0)
                x,first=layer_carrier[0],layer_carrier[1]
        if self.carrier_chain and self.te_final:
            x=self.mark('final-norm',self.final_norm(torch.stack((x,first)))[0])
        else:x=self.mark('final-norm',self.norm(x,'ln_out'))
        logits=F.linear(x,self.w('head.weight'))
        self.mark('logits',logits)
        return logits.transpose(0,1).contiguous() if self.aligned else logits

    def native_gradients(self,selected=None):
        result={}
        for name,param in self.weights.items():
            key=name.replace('__','.')
            if key.startswith('blocks.0.att.v') and key.rsplit('.',1)[-1] in ('v0','v1','v2'):continue
            if selected is not None and key not in selected:continue
            result[key]=param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param).cpu()
        return result
