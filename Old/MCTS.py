from math import sqrt

import numpy as np


class MCTS:

    def __init__(self, eval_method, simulations_per_turn, learning, c_puct=3, alpha=1):
        self.eval_method = eval_method
        self.simulations_per_turn = simulations_per_turn
        self.c_puct = c_puct

        self.alphas = np.full((200,), alpha, dtype=np.float32)
        self.idxs = np.arange(200)
        self.choose_move = self._choose_move_exploratory if learning else self._choose_move_greedy
        self.root_node = None

    def _choose_move_exploratory(self, game):
        if self.root_node is None:
            self.root_node = Node(game.copy(), self.eval_method)
        root = self.root_node

        # Add Dirichlet noise as in the paper
        root.P = (
            0.75 * root.P +
            0.25 * np.random.dirichlet(self.alphas[:game.num_moves])
        )

        for _ in range(self.simulations_per_turn):
            self._run_simulation(root)

        root.mcts_probs = (root.N - root.game.legal_moves) / (root.sumN - root.num_legal_moves)
        return np.random.choice(self.idxs[root.game.num_moves], p=root.mcts_probs)

    def _choose_move_greedy(self, game):
        if self.root_node is None:
            self.root_node = Node(game.copy(), self.eval_method)

        for _ in range(self.simulations_per_turn):
            self._run_simulation(self.root_node)

        return np.argmax(self.root_node.N)

    def done_move(self, move):
        if self.root_node is not None:
            self.root_node = self.root_node.nodes.get(move)

    def reset(self):
        self.root_node = None

    def _run_simulation(self, node):
        if isinstance(node, GameOverNode):
            return np.roll(node.V, 1)

        # Division by zero is ok since N is a np.ma.array
        Q = node.W[:, 0] / node.N
        over_N = sqrt(node.sumN) / (1 + node.N)
        U = Q + self.c_puct * node.P * over_N
        move = np.ma.argmax(U)

        next_node = node.nodes.get(move)
        if next_node is None:
            next_game = node.game.copy()
            res = next_game.do_move(move)
            next_node = Node(next_game, self.eval_method) if res is None else GameOverNode(res)
            node.nodes[move] = next_node
            next_v = np.roll(next_node.V, 1)
        else:
            next_v = self._run_simulation(next_node)

        node.W[move] += next_v
        node.N[move] += 1
        node.sumN += 1
        return np.roll(next_v, 1)


class Node:

    def __init__(self, game, eval_method):
        self.game = game
        self.nodes = {}

        P, V = eval_method(game)
        P = P.flatten() * game.legal_moves
        P *= 1 / np.sum(P)

        self.P = P
        self.V = V.flatten()
        self.N = np.ma.array(game.legal_moves, np.uint16)
        self.W = np.zeros((game.num_moves, game.num_players), np.float32)
        self.num_legal_moves = np.sum(game.legal_moves)
        self.sumN = self.num_legal_moves


class GameOverNode:
    def __init__(self, V):
        self.V = V

