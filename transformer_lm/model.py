import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
#from jax import random
from jax import lax
#import numpy as np
import numpy as np

def softmax(x, axis=-1):
	r"""Softmax function.

	Computes the function which rescales elements to the range :math:`[0, 1]`
	such that the elements along :code:`axis` sum to :math:`1`.

	.. math ::
		\mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

	Args:
		axis: the axis or axes along which the softmax should be computed. The
		softmax output summed across these dimensions should sum to :math:`1`.
		Either an integer or a tuple of integers.
	"""
	unnormalized = jnp.exp(x - lax.stop_gradient(x.max(axis, keepdims=True)))
	return unnormalized / unnormalized.sum(axis, keepdims=True)

def scaled_dot_product_att(Q, K, V, mode='train', mask=None):
	dk = Q.shape[-1]
	dv = V.shape[-1]

	QK = jnp.matmul(Q, jnp.transpose(K, axes=(0, 2, 1))) / jnp.sqrt(dk)
	if mask is not None:
		QK = jnp.multiply(QK, mask)
	attn = softmax(QK, axis=-1)
	out = jnp.matmul(QK, V)
	return out, attn

def multihead_attention(Q, K, V, WQs, WKs, WVs, Wout, num_heads, mode='train', mask=None):
	
	q_size, k_size, v_size = Q.shape[0],K.shape[0],V.shape[0]
	dk = Q.shape[-1]
	dv = V.shape[-1]

	res = Q

	# layer norm

	Qs = jnp.matmul(Q, WQs).reshape((q_size, num_heads, dk))
	Ks = jnp.matmul(K, WKs).reshape((k_size, num_heads, dk))
	Vs = jnp.matmul(V, WVs).reshape((v_size, num_heads, dv))

	Qs, Ks, Vs = jnp.transpose(Qs, axes=(1, 0, 2)), jnp.transpose(Ks, axes=(1, 0, 2)), jnp.transpose(Vs,axes=(1, 0, 2)) # (num_heads, seq_len, dx)
	
	out, attn = scaled_dot_product_att(Qs, Ks, Vs)

	mask = jnp.expand_dims(mask, axis=0)

	return out, attn

num_heads = 6
seq_len = 12
dk = 128
dv = 128
#hid_size = 256
in_feats = 128
bs = 2

WQs = np.random.rand(in_feats, num_heads * dk)
WKs = np.random.rand(in_feats, num_heads * dk)
WVs = np.random.rand(in_feats, num_heads * dv)

Q = np.random.rand(seq_len, in_feats)
K = np.random.rand(seq_len, in_feats)
V = np.random.rand(seq_len, in_feats)
mask = np.tril(np.ones((seq_len,seq_len)))

out, attn = multihead_attention(Q, K, V, WQs, WKs, WVs, None, num_heads, mode='train', mask=mask)

print(attn.shape)
#print(mask)
#print(scaled_dot_product_att(Q, K, V, mask))