import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
#from jax import random
from jax import lax
#import numpy as np
import numpy as np
from layers import layer_norm, multihead_attention, ff_block

def encoder_block(inputs, params, training=True):
    
    params_mha = params['mha']
    params_layer_norm_1 = params['layer_norm_1']
    params_layer_norm_2 = params['layer_norm_2']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs, params_mha, training=training)
    mha_res = layer_norm(mha + inputs[0], params_layer_norm_1, training=training)

    out = ff_block(mha_res, params_ff_block, training=training)
    out_res = layer_norm(out + mha_res, params_layer_norm_2, training=training)

    return out_res

def decoder_block(inputs, params, training=True):
    
    params_mha_1 = params['mha_1']
    params_mha_2 = params['mha_2']

    params_layer_norm_1 = params['layer_norm_1']
    params_layer_norm_2 = params['layer_norm_2']
    params_layer_norm_3 = params['layer_norm_3']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs, params_mha_1, training=training)
    mha_res = layer_norm(mha + inputs[0], params_layer_norm_1, training=training)

    out = ff_block(mha_res, params_ff_block, training=training)
    out_res = layer_norm(out + mha_res, params_layer_norm_2, training=training)

    return out_res