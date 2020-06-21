import numpy as np

from MCTS import MCTS


indexes = np.arange(13 * 13)


class Player:
    def __init__(self, id_=None):
        self.id = id_

    def choose_move(self, game):
        pass

    def done_move(self, move):
        pass

    def reset(self):
        pass


class Human(Player):
    def choose_move(self):
        raise NotImplementedError("Human players choose their own moves!")


class Random(Player):
    def choose_move(self, game):
        probs = game.legal_moves / np.sum(game.legal_moves)
        move = np.random.choice(indexes[:game.num_moves], p=probs)
        return move


class RandomMCTS(Player):

    def __init__(self, simulations_per_turn, max_rollout, learning=False, **kwargs):
        super().__init__(**kwargs)
        self.max_rollout = max_rollout

        self.rollout_player = Random()
        self.MCTS = MCTS(
            eval_method=lambda x: self._eval_state(x),
            simulations_per_turn=simulations_per_turn,
            learning=learning
        )

    def choose_move(self, game):
        return self.MCTS.choose_move(game)

    def done_move(self, move):
        self.MCTS.done_move(move)

    def reset(self):
        self.MCTS.reset()

    def _eval_state(self, game):
        probs = game.legal_moves * (1 / np.sum(game.legal_moves))
        rollout_game, rollout_player = game.copy(), self.rollout_player
        for turn in range(self.max_rollout):
            move = rollout_player.choose_move(rollout_game)
            results = rollout_game.do_move(move)
            if results is not None:
                return probs, np.roll(results, turn)
        return probs, np.zeros((game.num_players,),)


class PointsOnlyMCTS(Player):

    def __init__(self, simulations_per_turn=100, learning=False, **kwargs):
        super().__init__(**kwargs)
        self.MCTS = MCTS(
            eval_method=lambda x: self._eval_state(x),
            simulations_per_turn=simulations_per_turn,
            learning=learning
        )

    def choose_move(self, game):
        return self.MCTS.choose_move(game)

    def done_move(self, move):
        self.MCTS.done_move(move)

    def reset(self):
        self.MCTS.reset()

    def _eval_state(self, game):
        probs = game.legal_moves / np.sum(game.legal_moves)
        points = np.array([p[1, 0, 0] for p in game.player_layers])
        delta = points - np.min(points)
        delta = delta / (2 * np.max(delta) + 0.00001)
        points = points / 5 - 1
        scores = points + delta
        return probs, scores

