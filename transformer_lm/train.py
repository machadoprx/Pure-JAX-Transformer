import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from initializer import *
from tranformer_modules import *
from forward import *
from loss import *
from layers import *
from optimizer import *

def train_step(inputs, params, vocab_size):
	return lm_loss_fn(inputs, params, forward_transformer, vocab_size, training=True)

def train_loop(batched_inputs, params, vocab_size, epochs, lr):
	mask_ce = jnp.tril(jnp.ones((seq_len,seq_len)))
	for e in range(epochs):
		epoch_loss = 0.0
		# shuffle
		for batch, target in batched_inputs:
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0, 0, None], None, None)) \
																	([batch, target, target, mask_ce], params, vocab_size)
			epoch_loss += loss
			params = optimizer_sgd_tr(params, grads, ['eps', 'rate', 'mov_mean', 'mov_var', 'num_heads'], lr)
		print(f'Epoch: {e} - Loss: {epoch_loss/len(batched_inputs)}')
	return params


def debug():
	num_heads = 2
	seq_len = 4
	dk = 20
	dv = 20
	hid_size = 20
	vocab_size = 10
	epochs = 100
	lr = 0.01
	embed_size = 20
	#in_feats = 128
	bs = 2
	n_layers = 2
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	params = get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, n_layers, vocab_size, hid_size, rate_att=0.2, rate_ff=0.2, eps=1e-9)

	rng, subkey = jax.random.split(rng)

	Q = jnp.array([[[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]]])

	targets = jnp.array([[[4, 3, 2, 1], [4, 3, 2, 1]], [[4, 3, 2, 1], [4, 3, 2, 1]]])
	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))

	print(list(zip(Q, targets))[0])

	params = train_loop(list(zip(Q, targets)), params, vocab_size, epochs, lr)

debug()
