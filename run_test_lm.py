import numpy as np
import jax.numpy as jnp
import pickle
from src.vocabulary import Vocabulary
from src.datasets import *
from src.initializer import *
from src.train_lm import *
import sys
sys.path.append('src')

def main():

	with open('chess_db.txt', 'r') as f:
		corpus = f.readlines()[:20000]
	corpus = [line[:-1] for line in corpus]

	voc = pickle.load(open('voc.pkl', 'rb'))
	params = pickle.load(open('params.pkl', 'rb'))
	hyper_params = pickle.load(open('hyper_params.pkl', 'rb'))
	seq_len = hyper_params['max_len']
	#state = pickle.load(open('state.pkl', 'rb'))

	x = corpus[5]
	x = voc.encode(x)[:4]
	x[-1] = 1
	x = np.pad(x, (0, seq_len-len(x)), mode='constant')

	mask_input = x == 0
	mask_input = jnp.where(mask_input, -1e9, jnp.zeros((seq_len,seq_len)))
	
	print(voc.decode(list(np.array(x)))[:512])
	print(voc.decode(np.array(forward_test([x, mask_input], params, hyper_params)[0]))[:512])

if __name__ == '__main__':
    main()