from layers import layer_norm, multihead_attention, ff_block
from initializer import *

def encoder_block(inputs, params, layer, training=True):

    mha, _ = multihead_attention(inputs, params, f'encoder_{layer}_mha', training=training, causal=False)
    mha_res = layer_norm(mha + inputs[0], params, f'encoder_{layer}_ln_1', training=training)

    out = ff_block(mha_res, params, f'encoder_{layer}_ff_block', training=training)
    out = layer_norm(out + mha_res, params, f'encoder_{layer}_ln_2',training=training)

    return out

def decoder_block(inputs, params, layer, training=True):
    
    Q, K, V, enc_out, mask = inputs
    inputs_1 = [Q, K, V, mask]

    mha, _ = multihead_attention(inputs_1, params, f'decoder_{layer}_mha_1', training=training, causal=True)
    mha = layer_norm(mha + inputs_1[0], params, f'decoder_{layer}_ln_1', training=training)

    inputs_2 = [mha, enc_out, enc_out, mask]
    mha, _ = multihead_attention(inputs_2, params, f'decoder_{layer}_mha_2', training=training)
    mha = layer_norm(mha + inputs_2[0], params, f'decoder_{layer}_ln_2', training=training)

    out = ff_block(mha, params, f'decoder_{layer}_ff_block', training=training)
    out = layer_norm(out + mha, params, f'decoder_{layer}_ln_3', training=training)

    return out

'''
	embedding_enc = embed([input, pos_enc], params) * jnp.sqrt(hid_size)
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, out_enc, out_enc, mask_input], params, i, training=training)
	
	embedding_dec = embed([target, pos_enc], params) * jnp.sqrt(hid_size)
	out_dec = embedding_dec
	for i in range(n_layers):
		out_dec = decoder_block([out_dec, out_dec, out_dec, out_enc, mask_target], params, i, training=training)
'''