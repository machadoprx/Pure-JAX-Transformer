import jax.numpy as jnp
from layers import layer_norm, multihead_attention, ff_block
import jax
import jax.numpy as jnp
import numpy as np
#from jax import grad, jit, vmap, pmap
from jax import lax
from jax.nn import softmax, gelu
from initializer import *

def encoder_block(inputs, params, training=True):
    
    params_mha = params['mha']
    params_layer_norm_1 = params['ln_1']
    params_layer_norm_2 = params['ln_2']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs, params_mha, training=training)
    mha_res = layer_norm(mha + inputs[0], params_layer_norm_1, training=training)

    out = ff_block(mha_res, params_ff_block, training=training)
    out = layer_norm(out + mha_res, params_layer_norm_2, training=training)

    return out

def decoder_block(inputs, params, training=True):
    
    Q, K, V, enc_out, mask = inputs
    inputs_1 = [Q, K, V, mask]

    params_mha_1 = params['mha_1']
    params_mha_2 = params['mha_2']

    params_layer_norm_1 = params['ln_1']
    params_layer_norm_2 = params['ln_2']
    params_layer_norm_3 = params['ln_3']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs_1, params_mha_1, training=training)
    mha = layer_norm(mha + inputs_1[0], params_layer_norm_1, training=training)

    inputs_2 = [mha, enc_out, enc_out, None]
    mha, _ = multihead_attention(inputs_2, params_mha_2, training=training)
    mha = layer_norm(mha + inputs_2[0], params_layer_norm_2, training=training)

    out = ff_block(mha, params_ff_block, training=training)
    out = layer_norm(out + mha, params_layer_norm_3, training=training)

    return out