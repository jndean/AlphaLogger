import numpy as np
from tqdm import tqdm

import Game
import Player


def play_matches(players, num_matches=1, max_turns=100):
    num_players = len(players)
    scores = [0] * num_players
    board = Game.Board(num_players)

    for _ in tqdm(range(num_matches)):
        board.reset()
        any(player.reset() for player in players)
        moves = []
        for turn in range(max_turns):
            player = players[turn % num_players]
            move = player.choose_move(board)
            moves.append(move)
            result = board.do_move(move)
            any(p.done_move(move) for p in players)
            if result is not None:
                break
        else:
            continue

        winner = (turn + np.argmax(result)) % num_players
        scores[winner] += 1

    return scores


def debug_game(end_board, moves):
    n = end_board.num_players
    board = Game.Board(n)
    board.reset(player_positions=np.copy(Game.Board.corners[:n]))
    for move in moves:
        print(board.get_state()[1:4])
        print(Game.stringify_move(move, n))
        board.do_move(move)


if __name__ == '__main__':
    players = [
        # Player.RandomMCTS(id_="RandomMCTS100", simulations_per_turn=100, max_rollout=30),
        Player.PointsOnlyMCTS(id_="PointsMCTS", simulations_per_turn=100),
        Player.Random(id_="Random", ),

    ]
    num_matches = 10
    max_turns = 50

    scores = play_matches(players, num_matches, max_turns)

    for player, score in zip(players, scores):
        print(f"{player.id}: {score}")
    print(f"Unfinished: {num_matches - sum(scores)}")
