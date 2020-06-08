from collections import deque
from enum import Enum
from random import shuffle

import numpy as np


"""
Game state format:
    Variables in 5x5 2D planes, 5x5 board grid
    0: Plane marking trees of height 1 with a 1, -1 elsewhere
    1: Plane marking trees of height 2 with a 1, -1 elsewhere
    2: Plane marking trees of height 3 with a 1, -1 elsewhere
    3: Plane marking protesters with a 1, -1 elsewhere
    4+: For each player:
           Plane marking player position with 1, -1 elsewhere
           Const plane set to player score
           Const plane set to player num protesters
    
-1's are used so that 0's only appear in convolution padding -
I hope that makes the edges of the board more obvious?
Trees with a protester are marked in both the tree layer and the protester layer -
I wondered if this would make understanding choppable rows easier



Move indexing:
There are 13 possible motions a player can make, enumerated like so
_  _  0  _  _
_  1  2  3  _
4  5  6  7  8
_  9  10 11 _
_  _  12 _  _

After moving the actions are,
4 chops, 4 plants, num_opponents protests, and a pass
The action directions are enumerated like so
_ 0 _
1 _ 2
_ 3 _

So there are num_motions * num_actions moves, and they are enumerated to that the
actions changes most frequently

"""


def _within_bounds(x):
    return 0 <= x[0] < 5 and 0 <= x[1] < 5


class Board:

    corners = np.array([[0, 0], [0, 4], [4, 0], [4, 4]])
    directions = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
    motions = np.array([
        [-2, 0],
        [-1, -1], [-1, 0], [-1, 1],
        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -1], [1, 0], [1, 1],
        [2, 0]
    ])

    def __init__(self, num_players):
        self.num_players = num_players
        # 4 chops, 4 plants, n-1 protests, 1 pass
        self.num_actions = 8 + num_players
        # All combinations of one motion followed by one action
        self.num_moves = 13 * self.num_actions

        self.board, self.legal_moves = None, None
        self.player_positions, self.unoccupied = None, None
        self.num_unprotested_trees, self.player_layers = None, None

    def reset(self, player_positions=None):
        self.board = np.full((3 * self.num_players + 4, 5, 5), -1)
        self.player_layers = deque(np.full((3, 5, 5), -1) for _ in range(self.num_players))
        self.unoccupied = np.full((5, 5), True)
        self.num_unprotested_trees = 0

        np.random.shuffle(Board.corners)
        if player_positions is None:
            self.player_positions = deque(np.copy(Board.corners[:self.num_players]))
        else:
            self.player_positions = deque(player_positions[:])

        for (y, x), planes in zip(self.player_positions, self.player_layers):
            planes[0, y, x] = 1  # Player position
            planes[1] = 0  # Player score
            planes[2] = 1  # Player protesters
            self.unoccupied[y, x] = False

        self._update_legal_moves()

    def get_state(self):
        ret = np.empty_like(self.board)
        ret[:4] = self.board[:4]
        ret[4:] = np.vstack(self.player_layers)
        return ret

    def get_legal_moves(self):
        return self.legal_moves

    def _update_legal_moves(self):
        # Temporarily make the current spot empty
        unoccupied = self.unoccupied
        current_position = tuple(self.player_positions[0])
        unoccupied[current_position] = 1

        # First compute the legal motions
        new_positions = [tuple(pos) for pos in current_position + Board.motions]
        legal_motions = np.array([_within_bounds(p) and unoccupied[p] for p in new_positions])

        # Double moves require unoccupied transit spaces
        legal_motions[1] &= legal_motions[2] or legal_motions[5]
        legal_motions[3] &= legal_motions[2] or legal_motions[7]
        legal_motions[9] &= legal_motions[5] or legal_motions[10]
        legal_motions[11] &= legal_motions[7] or legal_motions[10]
        legal_motions[0] &= legal_motions[2]
        legal_motions[4] &= legal_motions[5]
        legal_motions[8] &= legal_motions[7]
        legal_motions[12] &= legal_motions[10]

        # Then compute the legal actions following legal motions
        legal_moves = np.full((self.num_moves,), False)
        block_idxs = range(0, self.num_moves, self.num_actions)
        tree_layer, protester_layer = self.board[2:4]
        for block_start, player_pos, legal_motion in zip(block_idxs, new_positions, legal_motions):
            if not legal_motion:
                continue
            block_end = block_start + self.num_actions - 1

            # For the adjacent square in each direction...
            for dir_idx, square in enumerate(player_pos + Board.directions):
                if not _within_bounds(square):
                    continue
                square = tuple(square)
                # ... can chop a mature tree if it's there
                legal_moves[block_start + dir_idx] = (
                        protester_layer[square] != 1 and tree_layer[square] == 1
                )
                # ... can plant a sapling if the square is empty
                legal_moves[block_start + dir_idx + 4] = unoccupied[square]

            # Can play protester if you have one and there's a suitable tree
            if self.player_layers[0][2][0][0] > 0 and self.num_unprotested_trees > 0:
                legal_moves[block_start + 8: block_end] = True
            # Must can pass if there are no available actions
            legal_moves[block_end] = not np.any(legal_moves[block_start: block_end])

        self.unoccupied[current_position] = 0  # Undo the temporary change
        self.legal_moves = legal_moves

    def do_move(self, move):
        # Move player
        position = self.player_positions[0]
        self.player_layers[0][0, position[0], position[1]] = -1
        position += Board.motions[move // self.num_actions]
        self.player_layers[0][0, position[0], position[1]] = 1
        self._grow()

        # Perform action
        action = move % self.num_actions
        if action < 4:
            self._chop(action)
        elif action < 8:
            self._plant(action - 4)
        else:
            self._protest(action - 8)

        # End move by changing player and caching the current legal moves
        self.player_layers.rotate(-1)
        self.player_positions.rotate(-1)
        self._update_legal_moves()

    def _grow(self):
        next_trees = np.copy(self.board[:3])

        def grow_square(y, x):
            if self.board[0, y, x] == 1:  # Grow sapling
                next_trees[0, y, x] = -1
                next_trees[1, y, x] = 1
            elif self.board[1, y, x] == 1:  # Grow mid-tree
                next_trees[1, y, x] = -1
                next_trees[2, y, x] = 1
                self.num_unprotested_trees += 1
            elif self.board[2, y, x] == 1:  # Mature tree spreads saplings
                for dir_y, dir_x in self.directions:
                    sq_y, sq_x = y + dir_y, x + dir_x
                    if _within_bounds((sq_y, sq_x)) and self.unoccupied[sq_y, sq_x]:
                        next_trees[0, sq_y, sq_x] = 1
                        self.unoccupied[sq_y, sq_x] = False

        player_y, player_x = self.player_positions[0]
        for _y in range(0, 5):
            grow_square(_y, player_x)
        for _x in range(0, 5):
            grow_square(player_y, _x)

        self.board[:3] = next_trees

    def _plant(self, dir_num):
        y, x = self.player_positions[0] + Board.directions[dir_num]
        self.board[0, y, x] = 1
        self.unoccupied[y, x] = False

    def _chop(self, dir_num):
        direction = Board.directions[dir_num]
        square = self.player_positions[0] + direction
        while _within_bounds(square) and self.board[2, square[0], square[1]] == 1:
            self.board[2, square[0], square[1]] = -1
            self.unoccupied[square[0], square[1]] = True
            self.player_layers[0][1] += 1
            if self.board[3, square[0], square[1]] == 1:  # Collect protester
                self.board[3, square[0], square[1]] = -1
                self.player_layers[0][2] += 1
            else:
                self.num_unprotested_trees -= 1
            square += direction

    def _protest(self, opponent_index):
        # Find closest tree to targeted opponent
        trees, protesters = self.board[2:4]
        opponent_y, opponent_x = self.player_positions[1 + opponent_index]
        min_dist, min_y, min_x = 999, None, None
        for y in range(5):
            for x in range(5):
                if trees[y, x] != 1 or protesters[y, x] == 1:
                    continue
                dy, dx = opponent_y - y, opponent_x - x
                dist = dy*dy + dx*dx
                if dist < min_dist:
                    min_dist, min_y, min_x = dist, y, x

        # Place protester and decrement protester count
        self.player_layers[0][2] -= 1
        protesters[min_y, min_x] = 1


if __name__ == '__main__':

    game = Board(2)
    game.reset()

    game.board[2, 3, 3] = 1
    game.unoccupied[3, 3] = False
    game.num_unprotested_trees = 1

    game._update_legal_moves()

    print(game.get_state())
    print(game.get_legal_moves().reshape((13, -1)))

    print(game.player_positions[0])
    print(Board.corners)
    game.player_positions[0] += np.array([1, 1])
    print(game.player_positions[0])
    print(Board.corners)
