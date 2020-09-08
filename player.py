import numpy as np

import core
import utilities


class Player:
    move_indices = np.arange(5*5*10)

    def __init__(self, name="No name"):
        self.name=name

    def choose_move(self, game):
        raise NotImplementedError()

    def done_move(self, move):
        pass

    def sync_with_game(self, game):
        pass


class HumanPlayer(Player):
    pass


def uniform_probs_from_game(game):
    legal_moves = game.get_legal_moves_array().flatten()
    probs = legal_moves / np.sum(legal_moves)
    return probs


tmp_P = np.ones(shape=(1, 5, 5, 10), dtype=np.float32) * (1/250)
def uniform_inference(game_array):
    # P = np.ones(shape=(game_array.shape[0], 5, 5, 10), dtype=np.float32) * (1/250)
    V = game_array[:, 0, 0, 5::3].astype(np.float32)
    return tmp_P.copy(order='C'), V  # .copy(order='C')


class RandomPlayer(Player):
    def choose_move(self, game):
        move_idx = np.random.choice(
            Player.move_indices, 
            p=uniform_probs_from_game(game)
        )
        return utilities.move_idx_to_tuple(move_idx)


class RandomMCTSPlayer(Player):

    def __init__(self, num_simulations=400, exploratory=False, **kwargs):
        super().__init__(**kwargs)
        self.num_simulations = num_simulations
        self.exploratory = exploratory
        self.mcts = core.MCTS(inference_method=uniform_inference)

    def sync_with_game(self, game):
        self.mcts.sync_with_game(game)

    def choose_move(self, game):
        move_idx = self.mcts.choose_move(
            num_simulations=self.num_simulations,
            exploratory=self.exploratory
        )
        return utilities.move_idx_to_tuple(move_idx)

    def done_move(self, move_tuple):
        self.mcts.done_move(utilities.move_tuple_to_idx(move_tuple))

