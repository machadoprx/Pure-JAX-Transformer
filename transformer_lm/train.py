import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from initializer import *
from tranformer_modules import *
from forward import *
from loss import *
from layers import *
from adamax import *
from tqdm import tqdm
import pickle

def train_step(inputs, params, hyper_params, vocab_size):
	return lm_loss_fn(inputs, params, hyper_params, forward_train, vocab_size)

def train_loop(batched_inputs, params, hyper_params, state, vocab_size, epochs, lr):
	step = 0
	for e in range(epochs):
		epoch_loss = 0.0
		batched_inputs = jax.random.permutation(jax.random.PRNGKey(np.random.randint(3000)), batched_inputs)
		for batch in tqdm(batched_inputs, total=len(batched_inputs)):
			x, target = batch[:, 0], batch[:, 1]
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0], None, None, None)) \
																	([x, target], params, hyper_params, vocab_size)
			epoch_loss += jnp.mean(loss)
			params, state = adamax(params, grads, state, step, lr=lr)
			#params = jax.tree_util.tree_map(lambda p, g: p - lr * jnp.mean(g, axis=0) , params, grads)
			step += 1
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

def get_ds_txt(voc, corpus, bs=8, min_len=8, max_len=128):
	corpus = corpus.split(' ')
	i = 0
	X = []
	y = []
	while i < len(corpus):
		rq = np.random.randint(min_len, max_len-2)
		ra = np.random.randint(min_len, max_len-2)
		if i + ra + rq >= len(corpus):
			break
		xq = [corpus[k] for k in range(i, rq+i)]
		i += rq
		xa = [corpus[k] for k in range(i, ra+i)]
		
		xq = ' '.join(xq)
		xa = ' '.join(xa)
		xq = jnp.array(voc.encode(xq))
		xa = jnp.array(voc.encode(xa))
		xq = jnp.pad(xq, (0, max_len-len(xq)), mode='constant')
		xa = jnp.pad(xa, (0, max_len-len(xa)), mode='constant')
		X.append(xq)
		y.append(xa)

	ds = list(zip(X, y)) 
	remain = len(ds) % bs
	ds = ds[:-remain]
	ds = jnp.asarray(ds).reshape((len(ds)//bs, bs, 2, max_len))

	return ds
		
def debug():
	num_heads = 6
	seq_len = 128
	dk = 512
	dv = dk
	hid_size = dk * 4
	
	epochs = 20
	lr = 2e-3
	ff_dim = hid_size * 4
	#in_feats = 128
	bs = 8
	n_layers = 4
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	#ds = get_sample_ds(size=16384, seq_len=seq_len, vocab_size=vocab_size, bs=bs)
	from vocabulary import Vocabulary
	with open('cleaned_corpus.txt', 'r') as f:
		corpus = f.readlines()[0]
	voc = Vocabulary(corpus)
	ds = get_ds_txt(voc, corpus, bs=bs, min_len=8, max_len=seq_len)
	vocab_size = len(voc.voc.keys())
	
	#print(ds[0][0][0])
	#print(ds.shape)
	#quit()

	params, hyper_params = get_transformer_params(rng, seq_len, dk, dv, hid_size, ff_dim, num_heads, n_layers, vocab_size)
	leaves, tree = jax.tree_util.tree_flatten(params)
	state = [jnp.zeros_like(p) for p in leaves]
	state = jax.tree_util.tree_unflatten(tree, state)
	state = {
		'mom':state,
		'inf':state
	}
	print(hyper_params)
	rng, subkey = jax.random.split(rng)

	#params = pickle.load(open('params.pkl', 'rb'))
	params = train_loop(ds, params, hyper_params, state, vocab_size, epochs, lr)
	
	f = open('params.pkl', 'wb'); pickle.dump(params,f); f.close()
	#f = open('hyper.pkl', 'wb'); pickle.dump(hyper_params,f); f.close()
	#params = pickle.load(open('data.obj', 'rb'))
	#print(params)

	seq_pred = []
	k = 79

	#x = [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 2, 0, 0, 0, 0, 0]
	x = ds[0][k][0]
	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	#print(mask_input)

	print(x)
	print(forward_test([x, mask_input], params, hyper_params))
	
debug()
