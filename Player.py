import abc

import numpy as np


class Player:
    def __init__(self, id_=None):
        self.id = id_

    @abc.abstractmethod
    def choose_move(self, game):
        pass


class HumanPlayer(Player):
    def choose_move(self):
        raise NotImplementedError("Human players choose their own moves!")


class RandomPlayer(Player):
    idxs = np.arange(13 * 13)

    def choose_move(self, game):
        probs = game.legal_moves / np.sum(game.legal_moves)
        move = np.random.choice(self.idxs[:game.num_moves], p=probs)
        return move
