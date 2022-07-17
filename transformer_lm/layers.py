import jax
import jax.numpy as jnp
#from jax import grad, jit, vmap, pmap
from jax import lax
from jax.nn import softmax, gelu
from initializer import *

def scaled_dot_product_att(inputs, params, training=True):
	Q, K, V, mask = inputs
	dk = Q.shape[-1]
	dv = V.shape[-1]

	QK = jnp.matmul(Q, jnp.transpose(K, axes=(0, 2, 1))) / jnp.sqrt(dk)
	if mask is not None:
		mask = jnp.expand_dims(mask, axis=0)
		QK = QK + mask # attn = attn.masked_fill(mask == 0, -1e9)

	attn = dropout(softmax(QK, axis=-1), {'rate':params['rate'], 'broadcast_dims':(), 'rng':params['rng']}, training=training)
	out = jnp.matmul(attn, V)
	return out, attn

def dropout(inputs, params, training=True):
	rate, broadcast_dims, rng = params['rate'], params['broadcast_dims'], params['rng']
	if rate <= 0:
		return inputs

	if rate >= 1.0:
		return jnp.zeros_like(inputs)

	keep_prob = 1.0 - rate

	if not training:
		return inputs
	else:
		broadcast_shape = list(inputs.shape)
		for dim in broadcast_dims:
			broadcast_shape[dim] = 1
		mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
		mask = jnp.broadcast_to(mask, inputs.shape)
		return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
	
def multihead_attention(inputs, params, training=True):
	
	Q, K, V, mask = inputs
	WQs, WKs, WVs, Wout, num_heads = params['WQs'], params['WKs'], params['WVs'], params['Wout'], params['num_heads']

	q_size, k_size, v_size = Q.shape[0],K.shape[0],V.shape[0]
	dk = Q.shape[-1]
	dv = V.shape[-1]

	Qs = jnp.matmul(Q, WQs).reshape((q_size, num_heads, dk))
	Ks = jnp.matmul(K, WKs).reshape((k_size, num_heads, dk))
	Vs = jnp.matmul(V, WVs).reshape((v_size, num_heads, dv))

	Qs, Ks, Vs = jnp.transpose(Qs, axes=(1, 0, 2)), jnp.transpose(Ks, axes=(1, 0, 2)), jnp.transpose(Vs,axes=(1, 0, 2)) # (num_heads, seq_len, dx)
	
	out, attn = scaled_dot_product_att([Qs, Ks, Vs, mask], params, training=training) # (num_heads, seq_len, dx), (num_heads, seq_len, seq_len)

	out = jnp.transpose(out, axes=(1, 0, 2)) # (seq_len, num_heads, dx)
	out = out.reshape(q_size, num_heads * dk) # (seq_len, num_heads * dx)

	out_proj = jnp.matmul(out, Wout)

	return out_proj, attn

def layer_norm(inputs, params, training=True):
	
	gamma, beta, mov_mean, mov_var, eps = params['gamma'], params['beta'], params['mov_mean'], params['mov_var'], params['eps']

	mean = inputs.mean(axis=-1, keepdims=True)
	var = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)

	if training:
		out = (inputs-mean)/jnp.sqrt(var + eps)
		params['mov_mean'] = (mov_mean * 0.9) + (mean * 0.1)
		params['mov_var'] = (mov_var * 0.9) + (var * 0.1)
	else:
		out = (inputs-mov_mean)/jnp.sqrt(mov_var + eps)

	out = (out * gamma) + beta

	return out

def ff_block(inputs, params, training=True):
	W1, W2 = params['W1'], params['W2']
	b1, b2 = params['b1'], params['b2']

	hid = gelu(jnp.matmul(inputs, W1) + b1)
	hid = dropout(hid, {'rate':params['rate'], 'broadcast_dims':(), 'rng':params['rng']}, training=training)
	out = gelu(jnp.matmul(hid, W2) + b2)
	return out

'''def test():
	num_heads = 6
	seq_len = 5
	dk = 128
	dv = 128
	hid_size = 128
	#in_feats = 128
	bs = 2
	rnd_range = 1 / hid_size ** 0.5
	rng = jax.random.PRNGKey(0)

	params_mha = get_mha_params(rng, dk, dv, hid_size, num_heads, 0.2)
	params_ff_block = get_ff_block_params(rng, hid_size, hid_size, 0.2)
	params_ln = get_ln_params(seq_len, eps=1e-9)

	Q = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)
	K = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)
	V = jax.random.uniform(rng, (seq_len, dk), minval=-rnd_range, maxval=rnd_range)

	mask = None

	inputs = [Q, K, V, mask]

	out, _ = multihead_attention(inputs, params_mha, training=True)
	print(out.shape)
	out = layer_norm(out + Q, params_ln, training=True)
	print(out.shape)
	out = ff_block(out, params_ff_block, training=True)
	print(out.shape)'''

#test()