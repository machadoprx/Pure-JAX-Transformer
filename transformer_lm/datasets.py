import numpy as np
import jax.numpy as jnp

def get_simple_ds(size=2048, seq_len=12, vocab_size=300, bs=8):
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

def get_ds_chess_mov_lvl(voc, corpus, bs=8, min_len=8, max_len=64):

	i = 0
	X = []
	y = []
	while i < len(corpus):
		game = corpus[i].split(' ')
		seq_len = len(game)
		if seq_len < min_len:
			i += 1
			continue
		
		moves = []
		start = 0
		query = np.random.randint(1, seq_len//2)
		while start < seq_len:
			moves.append(
				game[start:query]
			)
			start += query
			query = np.random.randint(start, seq_len)

		xt = []
		yt = []
		if len(moves) % 2 == 0:
			m = 0
			for k in range(len(moves)//2):
				
				m += 2

		else:
			pass
		

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