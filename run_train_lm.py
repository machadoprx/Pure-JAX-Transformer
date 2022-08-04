import numpy as np
import jax
import pickle
from src.vocabulary import Vocabulary
from src.datasets import *
from src.initializer import *
from src.train_lm import *

def main():
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
		corpus = f.readlines()
		corpus = corpus[:50000]
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

if __name__ == '__main__':
    main()