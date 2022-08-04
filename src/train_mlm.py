import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from src.initializer import *
from src.tranformer_modules import *
from src.forward_mlm import *
from src.loss import *
from src.layers import *
from src.adamax import *
from tqdm import tqdm
import pickle

def train_step(inputs, params, hyper_params, vocab_size):
	return mlm_loss_fn(inputs, params, hyper_params, forward_train, vocab_size)

def train_loop(batched_inputs, batched_inputs_val, params, hyper_params, state, voc, vocab_size, epochs, lr,seq_len):
	
	step = 0
	e = 0
	patience = 10
	mask_ratio = 0.15
	early_stop_flag = 0
	old_loss = float('inf')

	while early_stop_flag < patience:
		e += 1
		epoch_loss = 0.0
		batched_inputs = jax.random.permutation(jax.random.PRNGKey(np.random.randint(3000)), batched_inputs)
		for batch in tqdm(batched_inputs, total=len(batched_inputs)):
			x, target = batch[:, 0], batch[:, 1]
			mask = np.random.rand(*x.shape) < mask_ratio
			mask_skip = np.logical_or(np.logical_or(x == voc.voc['<CLS>'], x == voc.voc['<SEP>']), x == voc.voc['<PAD>'])
			mask_skip = np.logical_not(mask_skip)
			mask = np.logical_and(mask, mask_skip)
			x = np.where(mask, voc.voc['<MASK>'], x)
			loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0], None, None, None)) \
																	([x, target], params, hyper_params, vocab_size)
			epoch_loss += jnp.mean(loss)
			lr = lr_schedule(hyper_params['hid_size'], step)
			params, state = adamax(params, grads, state, step, lr=lr)
			step += 1

		val_loss = 0.
		for batch in tqdm(batched_inputs_val, total=len(batched_inputs_val)):
			x, target = batch[:, 0], batch[:, 1]
			mask = np.random.rand(*x.shape) < mask_ratio
			mask_skip = np.logical_or(np.logical_or(x == voc.voc['<CLS>'], x == voc.voc['<SEP>']), x == voc.voc['<PAD>'])
			mask_skip = np.logical_not(mask_skip)
			mask = np.logical_and(mask, mask_skip)
			x = np.where(mask, voc.voc['<MASK>'], x)
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
