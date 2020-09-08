
import numpy as np
from tqdm import tqdm

import core
import player


def play_matches(players, num_matches=1, max_turns=100):
    num_players = len(players)
    scores = [0] * num_players

    for _ in tqdm(range(num_matches)):
        game = core.LoggerState()
        any(p.sync_with_game(game) for p in players)
        for turn in range(max_turns):
            player = players[turn % len(players)]
            move = player.choose_move(game)
            winner = game.do_move(*move)
            any(p.done_move(move) for p in players)
            if winner is not None:
                scores[winner] += 1
                break

    return scores


def speed_test():
    from multiprocessing import Pool
    from time import time

    n_threads = 8
    t = time()
    with Pool(n) as pool:
        results = pool.map(speed_test, [None] * n_threads)
    t = time() - t
    simulations = sum(results)
    print(f'{simulations / t: 0.1f} simulations per second')

def speed_job(*args):
    n = 25
    s = 800
    P = player.RandomMCTSPlayer(name="R2", num_simulations=s)

    sims = 0
    for _ in range(n):
        game = core.LoggerState()
        P.sync_with_game(game)
        while 1:
            move = P.choose_move(game)
            sims += s
            P.done_move(move)
            if game.do_move(*move) is not None:
                break
    return sims


if __name__ == '__main__':

    players = [
        player.RandomMCTSPlayer(name="R2", num_simulations=500),
        player.RandomPlayer(name="R1"),
    ]
    num_matches = 100
    max_turns = 50

    scores = play_matches(players, num_matches, max_turns)

    for player, score in zip(players, scores):
        print(f"{player.name}: {score}")
