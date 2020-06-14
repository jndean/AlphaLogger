from math import sqrt

import numpy as np


class Node:
    c_puct = 3

    def __init__(self, game, eval_method):
        self.game = game
        self.eval_method = eval_method
        self.nodes = {}

        P, V = eval_method(game)
        P *= game.legal_moves
        P *= 1 / np.sum(P)

        self.P = P
        self.V = V
        self.N = np.ones((game.num_moves,), np.uint8)
        self.W = np.zeros((game.num_moves, game.num_players), np.float32)
        self.sumN = 0

    def run_simulation(self):
        if self.sumN:
            Q = self.W[:, 0] / self.N
            over_N = sqrt(self.sumN) / (1 + self.N)
            U = Q + self.c_puct * self.P * over_N
            move = np.argmax(U)
        else:
            move = np.argmax(self.game.legal_moves)

        node = self.nodes.get(move)
        if node is None:
            next_game = self.game.copy()
            result = next_game.do_move(move)
            node = Node(next_game, self.eval_method) if result is None else GameOverNode(result)
            self.nodes[move] = node
            next_v = np.roll(node.V, 1)
        else:
            next_v = node.run_simulation()

        self.W[move] += next_v
        self.N[move] += 1
        self.sumN += 1
        return np.roll(next_v, 1)

    def next_turn_exploratory(self):
        raise NotImplementedError()

    def next_turn_greedy(self):
        return np.argmax(self.N)


class GameOverNode:
    def __init__(self, V):
        self.V = V

    def run_simulation(self):
        return np.roll(self.V, 1)
