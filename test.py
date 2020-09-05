from time import time

import numpy as np

import core


def inference(inputs):
	P = np.ones(shape=(inputs.shape[0], 5, 5, 10), dtype=np.float32) * (1/250)
	V = inputs[:, 0, 0, 5::3].astype(np.float32)
	return P.copy(order='C'), V.copy(order='C')


mcts = core.MCTS(inference_method=inference)

game_state = core.LoggerState()
mcts.sync_with_game(game_state)

print(mcts.choose_move(num_simulations=100, exploratory=True))




# result = core.test_MCTS_selfplay(inference)

# print(result)







# state = core.LoggerState()

# s = time()
# state.test()
# t = time() - s

# print('Games per second:', 1_000_000 // t)