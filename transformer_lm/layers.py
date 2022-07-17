import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
import jax
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

def leaky_relu(x, negative_slope=1e-2):
	r"""Leaky rectified linear unit activation function.

	Computes the element-wise function:

	.. math::
		\mathrm{leaky\_relu}(x) = \begin{cases}
		x, & x \ge 0\\
		\alpha x, & x < 0
		\end{cases}

	where :math:`\alpha` = :code:`negative_slope`.
	"""
	return jnp.where(x >= 0, x, negative_slope * x)

def scaled_dot_product_att(Q, K, V, mask=None, training=True):
	dk = Q.shape[-1]
	dv = V.shape[-1]

	QK = jnp.matmul(Q / jnp.sqrt(dk), jnp.transpose(K, axes=(0, 2, 1))) 
	if mask is not None:
		QK = jnp.multiply(QK, mask) # attn = attn.masked_fill(mask == 0, -1e9)
	attn = softmax(QK, axis=-1)
	out = jnp.matmul(attn, V)
	return out, attn

def dropout(inputs, params, training=True):
	rate, broadcast_dims = params['rate'], params['broadcast_dims']
	if rate <= 0:
		return inputs

	if rate >= 1.0:
		return jnp.zeros_like(inputs)

	keep_prob = 1.0 - rate

	if not training:
		return inputs
	else:
		broadcast_shape = list(inputs.shape)
		for dim in self.broadcast_dims:
			broadcast_shape[dim] = 1
		mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
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
	
	mask = jnp.expand_dims(mask, axis=0)
	out, attn = scaled_dot_product_att(Qs, Ks, Vs, training=training, mask=mask) # (num_heads, seq_len, dx), (num_heads, seq_len, seq_len)

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

	hid = leaky_relu(jnp.matmul(inputs, W1) + b1, negative_slope=1e-4)
	hid = dropout(hid, params, training=training)
	out = leaky_relu(jnp.matmul(hid, W2) + b2, negative_slope=1e-4)
	return out

def test():
	num_heads = 6
	seq_len = 12
	dk = 128
	dv = 128
	hid_size = 128
	#in_feats = 128
	bs = 2

	WQs = np.random.rand(hid_size, num_heads * dk)
	WKs = np.random.rand(hid_size, num_heads * dk)
	WVs = np.random.rand(hid_size, num_heads * dv)
	Wout = np.random.rand(num_heads * dv, hid_size)

	W1 = np.random.rand(hid_size, hid_size)
	W2 = np.random.rand(hid_size, hid_size)

	gamma = np.ones((seq_len, 1))
	beta = np.zeros((seq_len, 1))
	mov_mean = np.zeros((seq_len, 1))
	mov_var = np.ones((seq_len, 1))
	eps = 1e-9

	Q = np.random.rand(seq_len, dk)
	K = np.random.rand(seq_len, dk)
	V = np.random.rand(seq_len, dv)
	mask = np.tril(np.ones((seq_len,seq_len)))

	inputs = [Q, K, V, mask]

	params = {
		'WQs':WQs,
		'WKs':WKs,
		'WVs':WVs,
		'Wout':Wout,
		'num_heads':num_heads,
		'gamma':gamma,
		'beta':beta,
		'mov_mean':mov_mean,
		'mov_var':mov_var,
		'eps':eps,
		'W1':W1,
		'W2':W2
	}

	out, _ = multihead_attention(inputs, params, training=True)
	print(out.shape)
	out = layer_norm(out, params, training=True)
	print(out.shape)
	out = ff_block(out, params, training=True)
	print(out.shape)

test()