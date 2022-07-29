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
from datasets import *
import pickle

def train_step(inputs, params, hyper_params, vocab_size):
	return lm_loss_fn(inputs, params, hyper_params, forward_train, vocab_size)

def train_loop(batched_inputs, params, hyper_params, state, voc, vocab_size, epochs, lr,seq_len):
	
	step = 0
	e = 0
	
	while True:
		e += 1
		epoch_loss = 0.0
		batched_inputs = jax.random.permutation(jax.random.PRNGKey(np.random.randint(3000)), batched_inputs)
		k = 0
		for batch in tqdm(batched_inputs, total=len(batched_inputs)):
			x, target = batch[:, 0], batch[:, 1]
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0], None, None, None)) \
																	([x, target], params, hyper_params, vocab_size)
			epoch_loss += jnp.mean(loss)
			lr = lr_schedule(hyper_params['hid_size'], step)
			params, state = adamax(params, grads, state, step, lr=lr)
			if k % 100 == 0:
				print(epoch_loss/k)
				x = batched_inputs[np.random.randint(0, len(batched_inputs))][0][0]
				mask_input = x == 0
				mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
				print(voc.decode(list(np.array(x))))
				print(voc.decode(list(np.array(forward_test([x, mask_input], params, hyper_params)[0]))))
				f = open('params.pkl', 'wb'); pickle.dump(params,f); f.close()
				f = open('state.pkl', 'wb'); pickle.dump(state,f); f.close()
			k += 1
			step += 1
		print(f'Epoch: {e + 1} - Loss: {epoch_loss/len(batched_inputs)}')
	return params
		
def debug():
	num_heads = 8
	seq_len = 128
	dk = 512
	dv = dk
	hid_size = dk
	
	epochs = 60
	lr = 5e-3
	ff_dim = hid_size * 4
	#in_feats = 128
	bs = 8
	n_layers = 1
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	#ds = get_sample_ds(size=16384, seq_len=seq_len, vocab_size=vocab_size, bs=bs)
	from vocabulary import Vocabulary
	with open('chess_db.txt', 'r') as f:
		corpus = f.readlines()[:12000]
	
	plain_corpus = []
	for line in corpus:
		plain_corpus.extend(line.split(' '))
	plain_corpus = ' '.join(plain_corpus)

	voc = Vocabulary(plain_corpus)
	ds = get_ds_chess_mov_lvl(voc, corpus, bs=bs, min_len=8, max_len=seq_len)
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
	#state = pickle.load(open('state.pkl', 'rb'))
	params = train_loop(ds, params, hyper_params, state, voc, vocab_size, epochs, lr, seq_len)
	

	#params = pickle.load(open('data.obj', 'rb'))
	#print(params)

	seq_pred = []
	k = 79

	#x = [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 2, 0, 0, 0, 0, 0]
	x = jnp.array(voc.encode('aureliano então'))
	x = jnp.pad(x, (0, seq_len-len(x)), mode='constant')
	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	#print(mask_input)

	print(voc.decode(list(np.array(x))))
	print(voc.decode(np.array(forward_test([x, mask_input], params, hyper_params)[0])))
	
debug()
