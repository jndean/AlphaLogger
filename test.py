from time import time

import numpy as np

import core
import player
# from model import create_model


num_moves = 5 * 5 * 10
S, P, V = core.self_play(player.uniform_inference, 10000, 50)

P = P.reshape((-1, num_moves))

print(S[:, 0, 0, 5::10].flatten())
print(S.shape, P.shape, V.shape)

model = create_model(
	input_shape=S[0].shape,
	num_moves=num_moves,
	num_players=core.num_players,
	num_resnet_blocks=10,
	conv_filters=64,
	value_dense_neurons=64,
)

model.fit(
	x=S,
	y={'policy_output': P, 'value_output': V}
)