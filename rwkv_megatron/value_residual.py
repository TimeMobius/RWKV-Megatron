"""Lossless views of SP's full-sequence, local-head value cache."""


def pack_sharded_value(value, tp_size):
    """Fold [S, B, H/TP] into [S/TP, B, H] for the pipeline carrier.

    The packed slot is an opaque byte layout, not token-aligned hidden states.
    Always restore [S, B, H/TP] before any per-token or per-sample computation.
    Contiguous projection outputs need no allocation or permutation here.
    """
    sequence, batch, channels = value.shape
    return value.reshape(sequence // tp_size, batch, channels * tp_size)


def unpack_sharded_value(packed, tp_size):
    """Restore the local heads over the complete sequence without communication."""
    sequence, batch, channels = packed.shape
    return packed.reshape(sequence * tp_size, batch, channels // tp_size)
