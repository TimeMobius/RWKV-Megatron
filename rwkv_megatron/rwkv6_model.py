from typing import Dict, List

import torch
from torch import Tensor
import torch.distributed

from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt import GPTModel
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import divide

from einops import rearrange

from .rwkv6_layer_specs import RWKV6ChannelMix, RWKV6TimeMix


class RWKV6Model(GPTModel):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.check_against_pretrained = False

        if self.pre_process:
            self.pre_norm = FusedLayerNorm(
                config, config.hidden_size, config.layernorm_epsilon
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        if self.check_against_pretrained:
            pretrained_model = self.ground_truth_model_closure()

            mcore_hiddens: List[Tensor] = []

            def mcore_hook(module, args, output):
                nonlocal mcore_hiddens
                mcore_hiddens += [
                    args[0] if isinstance(args, tuple) else args,
                    output[0] if isinstance(output, tuple) else output,
                ]

            hf_hiddens: List[Tensor] = []

            def hf_hook(module, args, output):
                nonlocal hf_hiddens
                hf_hiddens += [
                    t.transpose(0, 1)
                    for t in [
                        args[0] if isinstance(args, tuple) else args,
                        output[0] if isinstance(output, tuple) else output,
                    ]
                ]

            for m in [
                self.pre_norm,
                self.decoder.layers[0].self_attention.linear_key,
                *[
                    m1
                    for layer in self.decoder.layers
                    for m1 in [
                        layer.self_attention.linear_key,
                        layer.self_attention,
                        layer.mlp,
                    ]
                ],
                self.decoder.final_layernorm,
                self.output_layer,
            ]:
                m.register_forward_hook(mcore_hook)

            for m in [
                pretrained_model.rwkv.blocks[0].pre_ln,
                pretrained_model.rwkv.blocks[0].attention.key,
                *[
                    m1
                    for layer in pretrained_model.rwkv.blocks
                    for m1 in [
                        layer.attention.key,
                        layer.attention,
                        layer.feed_forward,
                    ]
                ],
                pretrained_model.rwkv.ln_out,
                pretrained_model.head,
            ]:
                m.register_forward_hook(hf_hook)

        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
            decoder_input = self.pre_norm(decoder_input)

        result = super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            inference_params,
            packed_seq_params,
            extra_block_kwargs,
        )

        if self.check_against_pretrained:
            with torch.no_grad():
                pretrained_model(input_ids.cpu())
                assert len(mcore_hiddens) == len(hf_hiddens)
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                torch.save(
                    [mcore_hiddens, hf_hiddens],
                    f"debug/{torch.distributed.get_rank()}.pth",
                )
                for rank in range(torch.distributed.get_world_size()):
                    torch.distributed.barrier()
                    if rank == torch.distributed.get_rank():
                        for i, (mcore, hf) in enumerate(zip(mcore_hiddens, hf_hiddens)):
                            mcore = mcore.detach().cpu()
                            if mcore.numel() != hf.numel():
                                assert mcore.dim() == hf.dim()
                                # find tp dimension
                                tp_dim = -1
                                for d in range(mcore.dim()):
                                    if mcore.size(d) * tp_size == hf.size(d):
                                        assert (
                                            tp_dim == -1
                                        ), "Multiple TP dimensions found"
                                        tp_dim = d
                                    else:
                                        assert mcore.size(d) == hf.size(d)
                                assert tp_dim != -1, "TP dimension not found"
                                # select in tp dimension
                                hf = hf.reshape(
                                    *hf.shape[:tp_dim],
                                    tp_size,
                                    -1,
                                    *hf.shape[tp_dim + 1 :],
                                )
                                hf = hf.select(tp_dim, tp_rank)
                            diff = (mcore - hf).abs() / (hf.abs() + 1e-4)
                            print(
                                f"rank {rank} layer {i} error: max {diff.max()} mean {diff.mean()}"
                            )
                            if diff.max() > 1:
                                topk_diffs, topk_indices = torch.topk(diff.view(-1), 10)
                                topk_hf = hf[
                                    torch.unravel_index(topk_indices, diff.shape)
                                ]
                                topk_mcore = mcore[
                                    torch.unravel_index(topk_indices, diff.shape)
                                ]
                                for i in range(10):
                                    print(
                                        "    ",
                                        topk_indices[i],
                                        topk_diffs[i],
                                        topk_mcore[i],
                                        topk_hf[i],
                                    )
                    torch.distributed.barrier()
            exit(0)

        return result

    def _get_layers_start(self):
        local_layers = len(self.decoder.layers)

        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()
        vp_size = get_virtual_pipeline_model_parallel_world_size()
        if vp_size is not None:
            vp_rank = get_virtual_pipeline_model_parallel_rank()
            stage = vp_rank * pp_size + pp_rank
        else:
            stage = pp_rank

        layers_start = stage * local_layers

        # first stage check
        assert layers_start != 0 or self.pre_process
        # last stage check
        assert (
            layers_start + local_layers != self.config.num_layers or self.post_process
        )

        return layers_start

    @torch.no_grad
    def from_mobius_huggingface(self, pretrained_model, check_against_pretrained: bool):
        state_dict: Dict[str, Tensor] = dict(pretrained_model.state_dict())

        layers_start = self._get_layers_start()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        def tp(full_tensor: Tensor, dim=0):
            global_size = full_tensor.size(dim)
            local_size = divide(global_size, tp_size)
            return full_tensor.narrow(dim, local_size * tp_rank, local_size)

        if self.config.layernorm_zero_centered_gamma:
            norm_offset = -1
        else:
            norm_offset = 0

        if self.pre_process:
            self.embedding.word_embeddings.weight.copy_(
                tp(state_dict["rwkv.embeddings.weight"])
            )
            self.pre_norm.weight.copy_(
                state_dict["rwkv.blocks.0.pre_ln.weight"] + norm_offset
            )
            self.pre_norm.bias.copy_(state_dict["rwkv.blocks.0.pre_ln.bias"])

        if self.post_process:
            self.decoder.final_layernorm.weight.copy_(
                state_dict["rwkv.ln_out.weight"] + norm_offset
            )
            self.decoder.final_layernorm.bias.copy_(state_dict["rwkv.ln_out.bias"])
            self.output_layer.weight.copy_(tp(state_dict["head.weight"]))

        for i_local_layer, layer in enumerate(self.decoder.layers):
            layer: TransformerLayer

            i_layer = i_local_layer + layers_start
            prefix = f"rwkv.blocks.{i_layer}."
            hf_layer = lambda term: state_dict[prefix + term]
            hf_cmix = lambda term: state_dict[prefix + "feed_forward." + term]
            hf_tmix = lambda term: state_dict[prefix + "attention." + term]

            cmix: RWKV6ChannelMix = layer.mlp
            tmix: RWKV6TimeMix = layer.self_attention

            layer.input_layernorm.weight.copy_(hf_layer("ln1.weight") + norm_offset)
            layer.input_layernorm.bias.copy_(hf_layer("ln1.bias"))

            tmix.mix_coeff_input.copy_(hf_tmix("time_maa_x").flatten())
            tmix.bottleneck_token_shift.weight_down.copy_(
                rearrange(hf_tmix("time_maa_w1"), "d (e r) -> d e r", e=5)
            )
            tmix.bottleneck_token_shift.weight_up.copy_(hf_tmix("time_maa_w2"))
            for i_in_bias, name in enumerate("wkvrg"):
                tmix.bottleneck_token_shift.bias[i_in_bias].copy_(
                    hf_tmix("time_maa_" + name).flatten()
                )
            tmix.bottleneck_decay.weight_down.copy_(
                hf_tmix("time_decay_w1").unsqueeze(1)
            )
            tmix.bottleneck_decay.weight_up.copy_(hf_tmix("time_decay_w2").unsqueeze(0))
            tmix.bottleneck_decay.bias.copy_(tp(hf_tmix("time_decay").flatten()))
            tmix.core_attention.time_first.copy_(tp(hf_tmix("time_faaaa").flatten()))
            tmix.linear_receptance.weight.copy_(tp(hf_tmix("receptance.weight")))
            tmix.linear_key.weight.copy_(tp(hf_tmix("key.weight")))
            tmix.linear_value.weight.copy_(tp(hf_tmix("value.weight")))
            tmix.linear_gate.weight.copy_(tp(hf_tmix("gate.weight")))
            tmix.linear_proj.weight.copy_(tp(hf_tmix("output.weight"), 1))
            tmix.group_norm.weight.copy_(tp(hf_tmix("ln_x.weight")))
            tmix.group_norm.bias.copy_(tp(hf_tmix("ln_x.bias")))

            layer.pre_mlp_layernorm.weight.copy_(hf_layer("ln2.weight") + norm_offset)
            layer.pre_mlp_layernorm.bias.copy_(hf_layer("ln2.bias"))

            cmix.mix_coeffs[0].copy_(hf_cmix("time_maa_k").flatten())
            cmix.mix_coeffs[1].copy_(hf_cmix("time_maa_r").flatten())
            cmix.linear_up.weight.copy_(tp(hf_cmix("key.weight"), 0))
            cmix.linear_gate.weight.copy_(tp(hf_cmix("receptance.weight"), 0))
            cmix.linear_down.weight.copy_(tp(hf_cmix("value.weight"), 1))

        if check_against_pretrained:
            assert self.pre_process and self.post_process
            self.check_against_pretrained = True
            self.ground_truth_model_closure = lambda: pretrained_model
