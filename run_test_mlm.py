import numpy as np
import jax.numpy as jnp
import pickle
from src.vocabulary import *
from src.datasets import *
from src.initializer import *
from src.forward_mlm import *
import sys
sys.path.append('src')

def main():
	with open('chess_db.txt', 'r') as f:
		corpus = f.readlines()
	corpus = [line[:-1] for line in corpus]

	voc = pickle.load(open('voc.pkl', 'rb'))
	params = pickle.load(open('params.pkl', 'rb'))
	hyper_params = pickle.load(open('hyper_params.pkl', 'rb'))
	seq_len = hyper_params['max_len']
	#state = pickle.load(open('state.pkl', 'rb'))
	x = corpus[-5]
	x = voc.encode(x)
	x = np.pad(x, (0, seq_len-len(x)), mode='constant')
	mask = np.random.rand(*x.shape) < 0.15
	mask_skip = np.logical_or(np.logical_or(x == voc.voc['<CLS>'], x == voc.voc['<SEP>']), x == voc.voc['<PAD>'])
	mask_skip = np.logical_not(mask_skip)
	mask = np.logical_and(mask, mask_skip)
	print(voc.decode(list(np.array(x)))[:512])
	x = np.where(mask, voc.voc['<MASK>'], x)

	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	
	print(voc.decode(list(np.array(x)))[:512])
	topks = np.array(forward_test([x, mask_input], params, hyper_params)[1])
	topks = topks[1:-1, :]
	print(topks.shape)
	print(voc.decode(topks[:, 0])[:512])
	print(voc.decode(topks[:, 1])[:512])
	print(voc.decode(topks[:, 2])[:512])

	print(voc.decode(np.array(forward_test([x, mask_input], params, hyper_params)[0]))[:512])

if __name__ == '__main__':
	main()