
import numpy as np

import Game
import Player


def self_play_matches(player, num_players=2, num_samples=100, max_turns=100):

    board = Game.Board(num_players)
    board.reset()
    player.reset()
    states = np.empty((num_samples+max_turns,) + board.get_state().shape, dtype=np.int8)
    probs = np.empty((num_samples+max_turns, board.num_moves), dtype=np.float32)
    scores = np.empty((num_samples+max_turns, num_players), dtype=np.float32)

    last_reset = 0

    for i in range(num_samples + max_turns):
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


if __name__ == "__main__":

    player = Player.RandomMCTS(id_="Player", simulations_per_turn=50, max_rollout=30, learning=True)
    states, probs, scores = self_play_matches(player, 3, num_samples=100)
