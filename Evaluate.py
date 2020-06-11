import numpy as np

import Game
import Player


def play_matches(players, num_matches=1, max_turns=1000):
    num_players = len(players)
    scores = [0] * num_players
    board = Game.Board(num_players)

    for _ in range(num_matches):
        board.reset()
        moves = []
        for turn in range(max_turns):
            player = players[turn % num_players]
            move = player.choose_move(board)
            moves.append(move)
            result = board.do_move(move)
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
        Player.RandomPlayer("Random1"),
        Player.RandomPlayer("Random2"),
    ]
    num_matches = 100
    max_turns = 100

    scores = play_matches(players, num_matches, max_turns)

    for player, score in zip(players, scores):
        print(f"{player.id}: {score}")
    print(f"Unfinished: {num_matches - sum(scores)}")
