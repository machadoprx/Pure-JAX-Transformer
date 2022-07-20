from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from initializer import *
from tranformer_modules import *
from forward import *
from loss import *
from layers import *
from tqdm import tqdm
import pickle

def train_step(inputs, params, vocab_size):
	return lm_loss_fn(inputs, params, forward_transformer, vocab_size, training=True)

def train_loop(batched_inputs, params, seq_len, vocab_size, epochs, lr):
	mask_ce = jnp.tril(jnp.ones((seq_len,seq_len)))
	for e in range(epochs):
		epoch_loss = 0.0
		# shuffle
		for batch, target in tqdm(batched_inputs, total=len(batched_inputs)):
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0, 0, None], None, None)) \
																	([batch, target, target, mask_ce], params, vocab_size)
			epoch_loss += jnp.mean(loss)
			
			params = jax.tree_util.tree_map(lambda p, g: p - lr * jnp.mean(g, axis=0) if not isinstance(p, int) else p, params, grads)
		print(f'Epoch: {e + 1} - Loss: {epoch_loss/len(batched_inputs)}')
	return params

def batched_inference(inputs, params, vocab_size):
	pass

def get_sample_ds(size=2048, seq_len=12, vocab_size=300, bs=8):
	X = []
	y = []
	for _ in range(size):
		start = 300
		while start >= 280:
			start = np.random.randint(1, high=vocab_size)
		x = np.arange(start, start + seq_len, 1, dtype=int)
		target =  x[::-1]
		#print(x, target)
		#quit()
		X.append(x)
		y.append(target)
	ds = list(zip(X, y)) 
	ds = jnp.asarray(ds).reshape((size//bs, 2, bs, seq_len))
	return ds


def debug():
	num_heads = 1
	seq_len = 4
	dk = 20
	dv = 20
	hid_size = 20
	vocab_size = 300
	epochs = 4
	lr = 0.05
	embed_size = 20
	#in_feats = 128
	bs = 8
	n_layers = 1
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	ds = get_sample_ds(size=2048, seq_len=seq_len, vocab_size=vocab_size, bs=bs)

	#params = get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, n_layers, vocab_size, hid_size)

	rng, subkey = jax.random.split(rng)

	Q = jnp.array([[[1,2,3,4], [101,102,103,104]], [[7,8,9,10], [17,18,19,20]]])


	targets = jnp.array([[[4, 3, 2, 1], [104,103,102,101]], [[10,9,8,7], [20,19,18,17]]])
	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))

	print(list(zip(Q, targets))[0])

	#params = train_loop(ds, params, seq_len, vocab_size, epochs, lr)
	
	#f = open('data.obj', 'wb')
	#pickle.dump(params,f)
	#f.close()
	params = pickle.load(open('data.obj', 'rb'))

	seq_pred = []

	print(jnp.argmax(softmax(forward_transformer([jnp.array([13,14,15,16]), jnp.array([16, 15, 14, 0])], params, training=False), axis=-1), axis=-1))
	
debug()
