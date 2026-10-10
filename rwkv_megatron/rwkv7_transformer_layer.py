"""RWKV7 value-residual transport on upstream Megatron layers."""

import torch

from megatron.core.transformer.transformer_layer import TransformerLayer


class RWKV7TransformerLayer(TransformerLayer):
    """Carry layer-zero values alongside each pipeline hidden-state shard."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        input_ids=None,
        *,
        inference_params=None,
    ):
        if inference_context is None:
            inference_context = inference_params
        values = None
        if hidden_states.ndim == 4:
            values = hidden_states[1:]
            hidden_states = hidden_states[0]
        normalized, residual, state = self._run_input_layernorm(hidden_states)
        if values is not None:
            normalized = torch.cat((normalized.unsqueeze(0), values), dim=0)
        attention_output, bias = self.self_attention(
            normalized,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        values = attention_output[1:]
        hidden_states = self._apply_self_attn_bda_step((attention_output[0], bias), residual, state)
        hidden_states, context = self._run_cross_attention(
            hidden_states, context, context_mask, inference_context
        )
        hidden_states = self._forward_mlp(
            hidden_states,
            inference_context,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
            input_ids=input_ids,
        )
        return torch.cat((hidden_states.unsqueeze(0), values), dim=0), context
