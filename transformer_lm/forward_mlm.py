import jax.numpy as jnp
from layers import *
from tranformer_modules import *

def forward_train(inputs, params, hyper_params):
	
	input, mask_input = inputs
	
	seq_len = len(input)
	n_layers = hyper_params['num_layers']
	hid_size = hyper_params['hid_size']
	pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)

	embedding_enc = embed([input, pos_enc], params)
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, mask_input], params, hyper_params, i, training=True)
	
	out = jnp.matmul(out_enc, params['embed'])

	return out

def forward_test(inputs, params, hyper_params, top_k=3):
	
	input, mask_input = inputs

	seq_len = len(input)
	n_layers = hyper_params['num_layers']
	hid_size = hyper_params['hid_size']
	pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)
	embedding_enc = embed([input, pos_enc], params) 
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, mask_input], params, hyper_params, i, training=False)
	
	logits = jnp.matmul(out_enc, params['embed'])
	
	top_k_token_ids = jnp.argsort(logits, axis=-1)[:, -top_k:][:, ::-1]
	out = jnp.argmax(logits, axis=-1)
	return [out, top_k_token_ids]
