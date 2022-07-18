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
    params_layer_norm_1 = params['layer_norm_1']
    params_layer_norm_2 = params['layer_norm_2']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs, params_mha, training=training)
    mha_res = layer_norm(mha + inputs[0], params_layer_norm_1, training=training)

    out = ff_block(mha_res, params_ff_block, training=training)
    out = layer_norm(out + mha_res, params_layer_norm_2, training=training)

    return out

def decoder_block(inputs, params, training=True):
    
    dec_in, enc_out, mask = inputs
    inputs_1 = [dec_in, dec_in, dec_in, mask]

    params_mha_1 = params['mha_1']
    params_mha_2 = params['mha_2']

    params_layer_norm_1 = params['layer_norm_1']
    params_layer_norm_2 = params['layer_norm_2']
    params_layer_norm_3 = params['layer_norm_3']
    params_ff_block = params['ff_block']

    mha, _ = multihead_attention(inputs_1, params_mha_1, training=training)
    mha = layer_norm(mha + inputs_1[0], params_layer_norm_1, training=training)

    inputs_2 = [mha, enc_out, enc_out, None]
    mha, _ = multihead_attention(inputs_2, params_mha_2, training=training)
    mha = layer_norm(mha + inputs_2[0], params_layer_norm_2, training=training)

    out = ff_block(mha, params_ff_block, training=training)
    out = layer_norm(out + mha, params_layer_norm_3, training=training)

    return out

def test():
	num_heads = 6
	seq_len = 5
	dk = 128
	dv = 128
	hid_size = 128
	#in_feats = 128
	bs = 2
	rnd_range = 1 / hid_size ** 0.5
	rng = jax.random.PRNGKey(42)

	rng, params_mha = get_mha_params(rng, dk, dv, hid_size, num_heads, 0.2)
	rng, params_ff_block = get_ff_block_params(rng, hid_size, hid_size, 0.2)
	params_ln = get_ln_params(seq_len, eps=1e-9)

	Q = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)
	rng, _ = jax.random.split(rng)
	K = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)
	rng, _ = jax.random.split(rng)
	V = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)
	rng, _ = jax.random.split(rng)

	mask = None

	inputs = [Q, K, V, mask]

	out, _ = multihead_attention(inputs, params_mha, training=True)
	print(out.shape)
	out = layer_norm(out + Q, params_ln, training=True)
	print(out.shape)
	out = ff_block(out, params_ff_block, training=True)
	print(out.shape)

test()