import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit
from jax.nn import softmax, gelu
from functools import partial

@jax.jit
def embed(inputs, params):
	seq, pos_enc = inputs
	W = params['embed']
	hid_size = W.shape[0]
	out = jnp.stack([(W[:, x] * jnp.sqrt(hid_size)) + pos_enc[i] for x, i in zip(seq, range(len(seq)))], axis=0)
	return out

@partial(jax.jit, static_argnames=['training', 'causal', 'rate'])
def scaled_dot_product_att(inputs, rate=0.2, training=True, causal=False):
	Q, K, V, mask = inputs
	dim = Q.shape[-1] # dim model!?
	seq_len = Q.shape[-2]

	# (num_heads, len_q, dim) @ (num_heads,dim,len_k)
	QK = jnp.matmul(Q, jnp.transpose(K, axes=(0, 2, 1))) / jnp.sqrt(dim)

	if mask is not None:
		QK = jnp.where(mask, -1e9, QK)

	if causal == True:
		mask_causal = jnp.tril(jnp.ones((seq_len,seq_len)))
		mask_causal = mask_causal == 0
		QK = jnp.where(mask_causal, -1e9, QK)

	attn = dropout(softmax(QK, axis=-1), training=training, rate=rate)
	out = jnp.matmul(attn, V)
	return out, attn

@partial(jax.jit, static_argnames=['rate', 'training'])
def dropout(inputs, training=True, rate=0.2):
	broadcast_dims = ()
	seed = np.random.randint((1 << 63) - 1)
	rng = jax.random.PRNGKey(seed)
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

def multihead_attention(inputs, params, hyper_params, key, training=True, causal=False):
	
	params_mha = params[key]
	Q, K, V, mask = inputs
	WQs, WKs, WVs, Wout, num_heads, rate = params_mha['WQs'], params_mha['WKs'], params_mha['WVs'], params_mha['Wout'], hyper_params['num_heads'], hyper_params['rate']

	q_size, k_size, v_size = Q.shape[0],K.shape[0],V.shape[0]
	dk = Q.shape[-1]
	dv = V.shape[-1]

	Qs = jit(jnp.matmul)(Q, WQs).reshape((q_size, num_heads, dk // num_heads))
	Ks = jit(jnp.matmul)(K, WKs).reshape((k_size, num_heads, dk // num_heads))
	Vs = jit(jnp.matmul)(V, WVs).reshape((v_size, num_heads, dv // num_heads))

	Qs, Ks, Vs = jnp.transpose(Qs, axes=(1, 0, 2)), jnp.transpose(Ks, axes=(1, 0, 2)), jnp.transpose(Vs,axes=(1, 0, 2)) # (num_heads, seq_len, hid_dim)

	out, attn = scaled_dot_product_att([Qs, Ks, Vs, mask], rate=rate, training=training, causal=causal) # (num_heads, seq_len, hid_dim), (num_heads, seq_len, seq_len)
	out = jnp.transpose(out, axes=(1, 0, 2)) # (seq_len, num_heads, hid_dim // num_heads)
	out = out.reshape(q_size, dv) # (seq_len, num_heads * hid_dim)

	out_proj = jit(jnp.matmul)(out, Wout)

	return out_proj, attn

@partial(jax.jit, static_argnames=['key', 'training', 'eps'])
def layer_norm(inputs, params, key, training=True, eps = 1e-9):
	
	params_ln = params[key]
	gamma, beta = params_ln['gamma'], params_ln['beta']
	
	mean = inputs.mean(axis=(-1,-2), keepdims=True)
	var = ((inputs - mean) ** 2).mean(axis=(-1,-2), keepdims=True)

	out = (inputs-mean)/jnp.sqrt(var + eps)
	out = (out * gamma) + beta
	return out

@partial(jax.jit, static_argnames=['key', 'training', 'rate'])
def ff_block(inputs, params, key, rate=0.2, training=True):
	params_ff = params[key]
	W1, W2 = params_ff['W1'], params_ff['W2']
	b1, b2 = params_ff['b1'], params_ff['b2']

	hid = dropout(gelu(jnp.matmul(inputs, W1) + b1), training=training, rate=rate)
	return jnp.matmul(hid, W2) + b2