import numpy as np

import MCTS


class Player:
    def __init__(self, id_=None):
        self.id = id_

    def choose_move(self, game):
        pass

    def done_move(self, move):
        pass


class Human(Player):
    def choose_move(self):
        raise NotImplementedError("Human players choose their own moves!")


class Random(Player):
    idxs = np.arange(13 * 13)

    def choose_move(self, game):
        probs = game.legal_moves / np.sum(game.legal_moves)
        move = np.random.choice(self.idxs[:game.num_moves], p=probs)
        return move


class RandomMCTS(Player):
    idxs = np.arange(13 * 13)

    def __init__(self, simulations_per_turn, max_rollout, **kwargs):
        super().__init__(**kwargs)
        self.simulations_per_turn = simulations_per_turn
        self.max_rollout = max_rollout

        self.rollout_player = Random()
        self.mcts_root = None

    def choose_move(self, game):
        if self.mcts_root is None:
            root = MCTS.Node(
                game.copy(),
                lambda x: self.eval_state(x)
            )
        else:
            root = self.mcts_root
        for _ in range(self.simulations_per_turn):
            root.run_simulation()

        return root.next_turn_greedy()

    def done_move(self, move):
        if self.mcts_root is not None:
            self.mcts_root = self.mcts_root.nodes.get(move)

    def eval_state(self, game):
        probs = game.legal_moves * (1 / np.sum(game.legal_moves))
        rollout_game, rollout_player = game.copy(), self.rollout_player
        for turn in range(self.max_rollout):
            move = rollout_player.choose_move(rollout_game)
            results = rollout_game.do_move(move)
            if results is not None:
                return probs, np.roll(results, turn)
        return probs, np.full((game.num_players,), 0.5)





