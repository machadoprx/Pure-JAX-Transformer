from layers import layer_norm, multihead_attention, ff_block
from initializer import *

def encoder_block(inputs, params, hyper_params, layer, training=True):

    x, mask = inputs
    mha, _ = multihead_attention([x, x, x, mask], params, hyper_params, f'encoder_{layer}_mha', training=training, causal=False)
    mha = layer_norm(mha + x, params, f'encoder_{layer}_ln_1', training=training, eps=hyper_params['eps'])

    out = ff_block(mha, params, f'encoder_{layer}_ff_block', training=training, rate=hyper_params['rate'])
    out_ln = layer_norm(out + mha, params, f'encoder_{layer}_ln_2',training=training, eps=hyper_params['eps'])

    return out_ln

def decoder_block(inputs, params, hyper_params, layer, training=True):
    
    tgt, tgt_mask, memory, memory_mask = inputs
    inputs_1 = [tgt, tgt, tgt, tgt_mask]

    mha_1, _ = multihead_attention(inputs_1, params, hyper_params, f'decoder_{layer}_mha_1', training=training, causal=True)
    mha_1 = layer_norm(mha_1 + tgt, params, f'decoder_{layer}_ln_1', training=training, eps=hyper_params['eps'])

    inputs_2 = [mha_1, memory, memory, memory_mask]
    mha_2, _ = multihead_attention(inputs_2, params, hyper_params, f'decoder_{layer}_mha_2', training=training)
    mha_2 = layer_norm(mha_2 + mha_1, params, f'decoder_{layer}_ln_2', training=training, eps=hyper_params['eps'])

    out = ff_block(mha_2, params, f'decoder_{layer}_ff_block', training=training, rate=hyper_params['rate'])
    out = layer_norm(out + mha_2, params, f'decoder_{layer}_ln_3', training=training, eps=hyper_params['eps'])

    return out