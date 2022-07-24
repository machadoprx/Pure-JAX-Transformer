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
	return lm_loss_fn(inputs, params, forward_train, vocab_size)

def train_loop(batched_inputs, params, vocab_size, epochs, lr):
	for e in range(epochs):
		epoch_loss = 0.0
		batched_inputs = jax.random.permutation(jax.random.PRNGKey(np.random.randint(3000)), batched_inputs)
		for batch in tqdm(batched_inputs, total=len(batched_inputs)):
			x, target = batch[:, 0], batch[:, 1]
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0], None, None)) \
																	([x, target], params, vocab_size)
			epoch_loss += jnp.mean(loss)
			params = jax.tree_util.tree_map(lambda p, g: p - lr * jnp.mean(g, axis=0) if not isinstance(p, int) else p, params, grads)
		print(f'Epoch: {e + 1} - Loss: {epoch_loss/len(batched_inputs)}')
	return params

def get_sample_ds(size=2048, seq_len=12, vocab_size=300, bs=8):
	X = []
	y = []
	random_seq = 8
	for _ in range(size):
		start = float('inf')
		while start >= vocab_size - random_seq - 1:
			start = np.random.randint(3, high=vocab_size)
		
		x = np.arange(start, start + random_seq, 1, dtype=int)
		
		target =  x[::-1]
		
		x = np.concatenate([np.array([1]),x,np.array([2])], axis=-1)
		yi = np.concatenate([np.array([1]),target,np.array([2])], axis=-1)
		
		x = np.pad(x, (0, seq_len-len(x)), mode='constant')
		yi = np.pad(yi, (0, seq_len-len(yi)), mode='constant')

		X.append(x)
		y.append(yi)

	ds = list(zip(X, y)) 
	ds = jnp.asarray(ds).reshape((size//bs, bs, 2, seq_len))

	return ds


def debug():
	num_heads = 4
	seq_len = 16
	dk = 8
	dv = 8
	hid_size = 32
	vocab_size = 301
	epochs = 10
	lr = 0.1
	ff_dim = 32
	#in_feats = 128
	bs = 256
	n_layers = 2
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	ds = get_sample_ds(size=16384 * 2, seq_len=seq_len, vocab_size=vocab_size, bs=bs)

	params = get_transformer_params(rng, seq_len, dk, dv, hid_size, ff_dim, num_heads, n_layers, vocab_size)

	rng, subkey = jax.random.split(rng)

	#params = pickle.load(open('data.obj', 'rb'))
	params = train_loop(ds, params, vocab_size, epochs, lr)
	
	f = open('data.obj', 'wb'); pickle.dump(params,f); f.close()
	#params = pickle.load(open('data.obj', 'rb'))
	#print(params)

	seq_pred = []
	k = 15

	#x = [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 2, 0, 0, 0, 0, 0]
	x = ds[0][k][0]
	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))

	print(x)
	print(forward_test([x, mask_input], params))
	
debug()
