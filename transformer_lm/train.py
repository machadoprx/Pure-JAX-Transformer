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
import random
import pickle

def train_step(inputs, params, vocab_size):
	return lm_loss_fn(inputs, params, forward_transformer, vocab_size, training=True)

def train_loop(batched_inputs, params, seq_len, vocab_size, epochs, lr):
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

def batched_inference(inputs, params, vocab_size):
	pass

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

	#print(ds[0])
	#quit()
	return ds


def debug():
	num_heads = 4
	seq_len = 16
	dk = 32
	dv = 32
	hid_size = 32
	vocab_size = 301
	epochs = 8
	lr = 0.1
	ff_dim = 128
	#in_feats = 128
	bs = 64
	n_layers = 1
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	ds = get_sample_ds(size=16384, seq_len=seq_len, vocab_size=vocab_size, bs=bs)

	params = get_transformer_params(rng, seq_len, dk, dv, hid_size, ff_dim, num_heads, n_layers, vocab_size)

	rng, subkey = jax.random.split(rng)

	#params = train_loop(ds, params, seq_len, vocab_size, epochs, lr)
	
	#f = open('data.obj', 'wb')
	#pickle.dump(params,f)
	#f.close()
	params = pickle.load(open('data.obj', 'rb'))
	#print(params)

	seq_pred = []
	k = 19

	#print(ds[0][k])
	#print(ds[0][0])
	in_dec = np.zeros((seq_len), dtype=np.int64)
	in_dec[0] = 1
	#print(in_dec)

	mask_input = ds[0][k][0] == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	#print(mask_input)
	mask_target = ds[0][k][1] == 0
	mask_target = jnp.where(mask_target, -1e9, jnp.zeros((seq_len,seq_len)))
	print(ds[0][k][0])
	print(jnp.argmax(softmax(forward_transformer([ds[0][k][0], mask_input, [1] + [0 for i in range(len(ds[0][k][1])-1)], mask_target], params, training=False), axis=-1), axis=-1)[0])
	
debug()
