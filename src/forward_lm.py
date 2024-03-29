import jax.numpy as jnp
from src.layers import *
from src.tranformer_modules import *
from jax.nn import softmax

def forward_train(inputs, params, hyper_params):
	
	input, mask_input, target, mask_target = inputs
	
	seq_len = len(input)
	n_layers = hyper_params['num_layers']
	hid_size = hyper_params['hid_size']
	pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)

	embedding_enc = embed([input, pos_enc], params)
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, mask_input], params, hyper_params, i, training=True)
	
	embedding_dec = embed([target, pos_enc], params)
	out_dec = embedding_dec
	for i in range(n_layers):
		out_dec = decoder_block([out_dec, mask_target, out_enc, mask_input], params, hyper_params, i, training=True)

	out = jnp.matmul(out_dec, params['embed'])

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
	
	out_token_ids = [1]
	top_ks = []
	for k in range(seq_len):
		x = jnp.array(out_token_ids + [0 for _ in range(seq_len-len(out_token_ids))])
		mask_out = x == 0
		mask_out = jnp.where(mask_out, -1e9, jnp.zeros((seq_len,seq_len)))
		embedding_dec = embed([x, pos_enc], params)
		out_dec = embedding_dec
		for i in range(n_layers):
			out_dec = decoder_block([out_dec, mask_out, out_enc, mask_input], params, hyper_params, i, training=False)
		out = jnp.matmul(out_dec, params['embed'])
		logits = softmax(out[k], axis=-1)
		out_token_id = jnp.argsort(logits)[-top_k:][::-1]
		top_ks.append(out_token_id)
		out_token_ids.append(out_token_id[0])
		if out_token_id[0] == 2: # id for EOSE
			break

	return jnp.array(out_token_ids), jnp.array(top_ks)