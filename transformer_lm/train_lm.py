import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from initializer import *
from tranformer_modules import *
from forward_lm import *
from loss import *
from layers import *
from adamax import *
from vocabulary import Vocabulary
from tqdm import tqdm
from datasets import *
import pickle

def train_step(inputs, params, hyper_params, vocab_size):
	return lm_loss_fn(inputs, params, hyper_params, forward_train, vocab_size)

def train_loop(batched_inputs, batched_inputs_val, params, hyper_params, state, voc, vocab_size, epochs, lr,seq_len):
	
	step = 0
	e = 0
	patience = 10
	early_stop_flag = 0
	old_loss = float('inf')

	while early_stop_flag < patience:
		e += 1
		epoch_loss = 0.0
		batched_inputs = jax.random.permutation(jax.random.PRNGKey(np.random.randint(3000)), batched_inputs)
		for batch in tqdm(batched_inputs, total=len(batched_inputs)):
			x, target = batch[:, 0], batch[:, 1]
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0], None, None, None)) \
																	([x, target], params, hyper_params, vocab_size)
			epoch_loss += jnp.mean(loss)
			lr = lr_schedule(hyper_params['hid_size'], step)
			params, state = adamax(params, grads, state, step, lr=lr)
			step += 1

		val_loss = 0.
		for batch in tqdm(batched_inputs_val, total=len(batched_inputs_val)):
			x, target = batch[:, 0], batch[:, 1]
			loss = vmap(train_step, in_axes=([0, 0], None, None, None)) \
										([x, target], params, hyper_params, vocab_size)
			val_loss += jnp.mean(loss)
		val_loss = val_loss/len(batched_inputs_val)
		if old_loss < val_loss:
			early_stop_flag += 1
		else:
			old_loss = val_loss
			early_stop_flag = 0
			f = open('params.pkl', 'wb'); pickle.dump(params,f); f.close()
			f = open('state.pkl', 'wb'); pickle.dump(state,f); f.close()
		print(f'Epoch: {e + 1} - Train Loss: {epoch_loss/len(batched_inputs)} - Val Loss: {val_loss}')

	return params
		
def debug_train():
	num_heads = 8
	seq_len = 256
	dk = 512
	dv = dk
	hid_size = dk
	
	epochs = 60
	lr = 5e-3
	ff_dim = hid_size * 4
	bs = 8
	n_layers = 4
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	with open('chess_db.txt', 'r') as f:
		corpus = f.readlines()[:5000]
	corpus = [line[:-1] for line in corpus]

	plain_corpus = []
	for line in corpus:
		plain_corpus.extend(line.split(' '))
	plain_corpus = ' '.join(plain_corpus)
	
	voc = Vocabulary(plain_corpus)
	ds = get_ds_chess_mov_lvl_lm(voc, corpus, bs=bs, min_len=8, max_len=seq_len)
	vocab_size = len(voc.voc.keys())
	
	ds_train = ds[:int(len(ds)*0.8)]
	ds_test = ds[int(len(ds)*0.8):]


	params, hyper_params = get_lm_params(rng, seq_len, hid_size, ff_dim, num_heads, n_layers, vocab_size)
	f = open('voc.pkl', 'wb'); pickle.dump(voc,f); f.close()
	f = open('hyper_params.pkl', 'wb'); pickle.dump(hyper_params,f); f.close()

	leaves, tree = jax.tree_util.tree_flatten(params)
	state = [jnp.zeros_like(p) for p in leaves]
	state = jax.tree_util.tree_unflatten(tree, state)
	state = {
		'mom':state,
		'inf':state
	}
	print(hyper_params)
	rng, subkey = jax.random.split(rng)

	params = train_loop(ds_train, ds_test, params, hyper_params, state, voc, vocab_size, epochs, lr, seq_len)

def debug_test():
	
	with open('chess_db.txt', 'r') as f:
		corpus = f.readlines()[:20000]
	corpus = [line[:-1] for line in corpus]

	voc = pickle.load(open('voc.pkl', 'rb'))
	params = pickle.load(open('params.pkl', 'rb'))
	hyper_params = pickle.load(open('hyper_params.pkl', 'rb'))
	seq_len = hyper_params['max_len']
	#state = pickle.load(open('state.pkl', 'rb'))

	x = corpus[4]
	x = voc.encode(x)[:4]
	x[-1] = 1
	x = np.pad(x, (0, seq_len-len(x)), mode='constant')

	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	
	print(voc.decode(list(np.array(x)))[:512])
	print(voc.decode(np.array(forward_test([x, mask_input], params, hyper_params)[0]))[:512])


debug_test()
