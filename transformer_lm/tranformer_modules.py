from layers import layer_norm, multihead_attention, ff_block
from initializer import *

def encoder_block(inputs, params, layer, training=True):

    x, mask = inputs
    mha, _ = multihead_attention([x, x, x, mask], params, f'encoder_{layer}_mha', training=training, causal=False)
    mha_res = layer_norm(mha + inputs[0], params, f'encoder_{layer}_ln_1', training=training)

    out = ff_block(mha_res, params, f'encoder_{layer}_ff_block', training=training)
    out = layer_norm(out + mha_res, params, f'encoder_{layer}_ln_2',training=training)

    return out

def decoder_block(inputs, params, layer, training=True):
    
    tgt, tgt_mask, memory, memory_mask = inputs
    inputs_1 = [tgt, tgt, tgt, tgt_mask]

    mha, _ = multihead_attention(inputs_1, params, f'decoder_{layer}_mha_1', training=training, causal=True)
    mha = layer_norm(mha + inputs_1[0], params, f'decoder_{layer}_ln_1', training=training)

    inputs_2 = [mha, memory, memory, memory_mask]
    mha, _ = multihead_attention(inputs_2, params, f'decoder_{layer}_mha_2', training=training)
    mha = layer_norm(mha + inputs_2[0], params, f'decoder_{layer}_ln_2', training=training)

    out = ff_block(mha, params, f'decoder_{layer}_ff_block', training=training)
    out = layer_norm(out + mha, params, f'decoder_{layer}_ln_3', training=training)

    return out