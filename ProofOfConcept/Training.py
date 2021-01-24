from random import randrange as rng

import numpy as np
from tqdm import tqdm

from Evaluate import play_matches
import Game
from Model import create_model
import Player


def self_play_matches(player, num_players=2, num_samples=100, max_turns=100, silent=False):

    board = Game.Board(num_players)
    board.reset()
    player.reset()
    states = np.empty((num_samples+max_turns,) + board.get_state().shape, dtype=np.int8)
    probs = np.empty((num_samples+max_turns, board.num_moves), dtype=np.float32)
    scores = np.empty((num_samples+max_turns, num_players), dtype=np.float32)

    last_reset = 0
    _tqdm = (lambda x: x) if silent else tqdm
    for i in _tqdm(range(num_samples + max_turns)):
        move = player.choose_move(board)

        states[i] = board.get_state()
        probs[i] = player.MCTS.root_node.mcts_probs

        result = board.do_move(move)
        player.done_move(move)

        if result is not None:
            # Retroactively set scores for the whole match
            turn_offset = (i - last_reset) % num_players
            for pid in range(num_players):
                scores[last_reset + pid: i + 1: num_players] = np.roll(result, turn_offset - pid)

            if i >= num_samples:
                break
            last_reset = i
            board.reset()
            player.reset()

    return states[:num_samples], probs[:num_samples], scores[:num_samples]


"""
Since playing games is more expensive than training, we exploit the symmetries of
the board and rules to augment data from played games.
The reflection symmetries are enumerated 
       0     1
   \   |   /
     \ | / 
  -----|------ 2
     / | \
   /   |   \
             3
rotations are enumerated by multiples of 90 degrees clockwise
"""


movement_permutations = [
    [0, 3, 2, 1, 8, 7, 6, 5, 4, 11, 10, 9, 12],  # Reflection 0
    [8, 11, 7, 3, 12, 10, 6, 2, 0, 9, 5, 1, 4],  # Reflection 1
    [12, 9, 10, 11, 4, 5, 6, 7, 8, 1, 2, 3, 0],  # Reflection 2
    [4, 1, 5, 9, 0, 2, 6, 10, 12, 3, 7, 11, 8],  # Reflection 3
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Rotation 0 (identity)
    [4, 9, 5, 1, 12, 10, 6, 2, 0, 11, 7, 3, 8],  # Rotation 1
    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # Rotation 2
    [8, 3, 7, 11, 0, 2, 6, 10, 12, 1, 5, 9, 4],  # Rotation 3
]

action_permutations = [
    [0, 2, 1, 3, 4, 6, 5, 7],  # Reflection 0
    [2, 3, 0, 1, 6, 7, 4, 5],  # Reflection 1
    [3, 1, 2, 0, 7, 5, 6, 4],  # Reflection 2
    [1, 0, 3, 2, 5, 4, 7, 6],  # Reflection 3
    [0, 1, 2, 3, 4, 5, 6, 7],  # Rotation 0 (identity)
    [1, 3, 0, 2, 5, 7, 4, 6],  # Rotation 1
    [3, 2, 1, 0, 7, 6, 5, 4],  # Rotation 2
    [2, 0, 3, 1, 6, 4, 7, 5],  # Rotation 3
]

state_transforms = [
    lambda x: np.flip(x, axis=1),
    lambda x: np.transpose(x[::-1,::-1, :], axes=(1, 0, 2)),
    lambda x: np.flip(x, axis=0),
    lambda x:  np.transpose(x, axes=(1, 0, 2)),
    lambda x: x,
    lambda x: np.rot90(x),
    lambda x: np.rot90(x, k=2),
    lambda x: np.rot90(x, k=3),
]


def transform_move(move, transform_idx, num_moves):
    x = np.zeros((num_moves,))
    x[move] = 1
    row_perm = movement_permutations[transform_idx]
    col_perm = action_permutations[transform_idx]
    x = x.reshape((13, -1))[row_perm][:, col_perm].flatten()
    return np.argmax(x)


def transform_tuple(state, probs, transform_idx):
    row_perm = movement_permutations[transform_idx]
    col_perm = action_permutations[transform_idx]
    out_probs = probs[row_perm][:, col_perm]
    out_state = state_transforms[transform_idx](state)
    return out_state, out_probs


def augment_and_sample(data, num_samples):
    data = data[np.random.choice(data.shape[0], num_samples, replace=False), ...]
    data = data.reshape((data.shape[0], 13, -1))
    for i in range(data.shape[0]):
        p_i = rng(8)
        row_perm = movement_permutations[p_i]
        col_perm = action_permutations[p_i]
        data[i] = data[i][row_perm][:, col_perm]
    return data

"""
def test_transforms():

    player = Player.Random('random')
    for _ in tqdm(range(10)):
        num_players = 2 + rng(3)
        game = Game.Board(num_players)
        game.reset()
        result = None
        while result is None:
            move = player.choose_move(game)
            next_game = game.copy()
            result = next_game.do_move(move)

            for transform_idx in range(8):
                move_t = transform_move(move, transform_idx, game.num_moves)
"""


if __name__ == "__main__":

    num_players = 2
    game = Game.Board(num_players)
    game.reset()

    model = create_model(
        input_shape=game.get_state().shape,
        num_moves=game.num_moves,
        num_players=num_players,
        num_resnet_blocks=20,
        conv_filters=256,
        value_dense_neurons=256,
        lr_momentum=0.9,
        l2_regularisation=0.0001
    )

    training_player = Player.AlphaLogger(
        id_="AL_training", model=model, simulations_per_turn=50, learning=True
    )
    evaluation_player = Player.AlphaLogger(
        id_="AL_evaluation", model=model, simulations_per_turn=50, learning=False
    )
    random_opponent = Player.RandomMCTS(id_="Random", simulations_per_turn=50, max_rollout=30)

    while 1:
        for _ in range(5):
            states, probs, scores = self_play_matches(training_player, num_players, num_samples=200)
            model.fit(states, [probs, scores], epochs=2)

        scores = play_matches([evaluation_player, random_opponent], num_matches=20, max_turns=50)
        print(f'Win rate: {scores[0]}/8')



