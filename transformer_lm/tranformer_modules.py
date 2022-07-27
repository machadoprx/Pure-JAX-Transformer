from layers import layer_norm, multihead_attention, ff_block
from initializer import *

def encoder_block(inputs, params, hyper_params, layer, training=True):

    x, mask = inputs
    ln = layer_norm(x, params, f'encoder_{layer}_ln_1', training=training, eps=hyper_params['eps'])
    mha, _ = multihead_attention([ln, ln, ln, mask], params, hyper_params, f'encoder_{layer}_mha', training=training, causal=False)
    mha = mha + x

    ln2 = layer_norm(mha, params, f'encoder_{layer}_ln_2',training=training, eps=hyper_params['eps'])
    out = ff_block(ln2, params, f'encoder_{layer}_ff_block', training=training, rate=hyper_params['rate'])
    out = out + mha

    return out

def decoder_block(inputs, params, hyper_params, layer, training=True):
    
    tgt, tgt_mask, memory, memory_mask = inputs

    ln1 = layer_norm(tgt, params, f'decoder_{layer}_ln_1', training=training, eps=hyper_params['eps'])
    mha_1, _ = multihead_attention([ln1, ln1, ln1, tgt_mask], params, hyper_params, f'decoder_{layer}_mha_1', training=training, causal=True)
    mha_1 = mha_1 + tgt
    
    ln2 = layer_norm(mha_1, params, f'decoder_{layer}_ln_2', training=training, eps=hyper_params['eps'])
    mha_2, _ = multihead_attention([ln2, memory, memory, memory_mask], params, hyper_params, f'decoder_{layer}_mha_2', training=training)
    mha_2 = mha_2 + mha_1

    ln3 = layer_norm(mha_2, params, f'decoder_{layer}_ln_3', training=training, eps=hyper_params['eps'])
    out = ff_block(ln3, params, f'decoder_{layer}_ff_block', training=training, rate=hyper_params['rate'])
    out = out + mha_2

    return out