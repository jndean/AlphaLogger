from time import time
from time import time

import numpy as np

import core
import player
from model import create_model


num_moves = 5 * 5 * 10
num_state_array_channels = core.num_players * 3 + 4


print('Building model...')
model = create_model(
	input_shape=(5, 5, num_state_array_channels),
	num_moves=num_moves,
	num_players=core.num_players,
	num_resnet_blocks=10,
	conv_filters=64,
	value_dense_neurons=64,
)

def infer(x):
	return tuple(model.predict(x))


print('Generating games...')
t = time()
S, P, V = core.self_play(infer, 2560, 50)
print(time() - t, 'seconds')

print('Training...')
model.fit(
	x=S,
	y={'policy_output': P, 'value_output': V},
)
