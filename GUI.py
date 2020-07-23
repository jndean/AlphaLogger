from collections import deque
import tkinter as tk
import tkinter.font as font

import numpy as np

from logger import LoggerState


class MatchGUI:

    def __init__(self, players):
        self.players = deque(players)
        self.num_players = len(players)

        self.game_state = LoggerState(self.num_players)

        self.num_actions = 10

        self.window, self.grid = None, None
        self.chosen_motion, self.chosen_action self.chosen_protester = None, None, None

        self.game_over = False
        self.input_lock = False
        self.create_gui()
        self.draw()
        self.window.mainloop()

    def create_gui(self):
        # Initialise GUI
        self.window = tk.Tk()
        layout = tk.Frame()
        grid_frame = tk.Frame(master=layout)
        self.grid = [[], [], [], [], []]
        for row_num in range(5):
            for col_num in range(5):
                label = tk.Button(
                    width=10, height=5, borderwidth=2, relief="groove", master=grid_frame
                )
                label.grid(row=row_num, column=col_num)
                self.grid[row_num].append(label)
        grid_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        self.message_box = tk.Label(master=layout)
        self.message_box.grid(row=2, column=0)

        def add_button(y, x, label, master, method, arg):
            button = tk.Button(
                text=label, width=3, height=1, master=master, command=lambda: method(arg)
            )
            button['font'] = tk.font.Font(size=20)
            button.grid(row=y, column=x)

        direction_frame = tk.Frame(master=layout)
        for i, ((y, x), label) in enumerate([
            ((0, 2), "↟"),
            ((1, 1), "↖"), ((1, 2), "↑"), ((1, 3), "↗"),
            ((2, 0), "↞"), ((2, 1), "←"), ((2, 2), "."), ((2, 3), "→"), ((2, 4), "↠"),
            ((3, 1), "↙"), ((3, 2), "↓"), ((3, 3), "↘"),
            ((4, 2), "↡")
        ]):
            add_button(y, x, label, direction_frame, self.set_motion, i)
        direction_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5)

        chop_frame = tk.Frame(master=layout)
        for i, ((y, x), label) in enumerate([
            ((0, 1), "↑"), ((1, 0), "←"), ((1, 2), "→"), ((2, 1), "↓")
        ]):
            add_button(y, x, label, chop_frame, self.set_action, i)
        tk.Label(text="CHOP", master=chop_frame).grid(row=1, column=1, padx=5, pady=5)
        chop_frame.grid(row=0, column=2)

        plant_frame = tk.Frame(master=layout)
        for i, ((y, x), label) in enumerate([
            ((0, 1), "↑"), ((1, 0), "←"), ((1, 2), "→"), ((2, 1), "↓")
        ]):
            add_button(y, x, label, plant_frame, self.set_action, i+4)
        tk.Label(text="PLANT", master=plant_frame).grid(row=1, column=1, padx=5, pady=5)
        plant_frame.grid(row=1, column=2)

        extras_frame = tk.Frame(master=layout)
        tk.Button(
            text="No Action", width=12, height=1, master=extras_frame,
            command=lambda: self.set_action(7 + self.num_players)
        ).pack()
        tk.Button(
            text="Protest", width=12, height=1, master=extras_frame,
            command=lambda: self.set_action(8)
        ).pack()
        extras_frame.grid(row=0, column=3, padx=25)

        layout.pack()

    def set_motion(self, idx):
        if self.input_lock:
            return
        self.input_lock = True
        self.chosen_motion = idx
        self.continue_game()
        self.input_lock = False

    def set_action(self, idx):
        if self.input_lock:
            return
        self.input_lock = True
        self.chosen_action = idx
        self.continue_game()
        self.input_lock = False

    def set_protester(self, xy):
        if self.input_lock:
            return
        self.chosen_protester = xy

    def continue_game(self):
        while not self.game_over:
            current_player = self.players[0]
            if isinstance(current_player, Player.Human):
                if self.chosen_motion is None or self.chosen_action is None:
                    return
                move = self.num_actions * self.chosen_motion + self.chosen_action
                self.chosen_motion, self.chosen_action = None, None
                if not self.board.legal_moves[move]:
                    self.print(f"Illegal move ({move})")
                    print(self.board.legal_moves.reshape((13, -1)))
                    return
            else:
                move = current_player.choose_move(self.board)

            message = f'"{current_player.id}" plays {Game.stringify_move(move, self.num_players)}'
            if not self.board.legal_moves[move]:
                message += ". That's illegal!"
            self.print(message)

            final_scores = self.board.do_move(move)
            for player in self.players:
                player.done_move(move)

            if final_scores is not None:
                self.draw()
                winner = self.players[np.argmax(final_scores)]
                self.print(f"{winner.id} wins!")
                self.game_over = True
                return

            self.players.rotate(-1)
            self.draw()

    def print(self, message):
        self.message_box['text'] = message

    def draw(self):
        board = self.game_state.get_state_array()
        for y in range(5):
            for x in range(5):
                if board[y, x, 0] == 1:
                    label = "1"
                elif board[y, x, 1] == 1:
                    label = "2"
                elif board[y, x, 3] == 1:
                    label = "3 :O"
                elif board[y, x, 2] == 1:
                    label = "3"
                else:
                    for i, player in enumerate(self.players):
                        if board[y, x, 4 + 3 * i] == 1:
                            label = f"{player.id}\n{board[y, x, 5 + 3 * i]} ({board[y, x, 6 + 3 * i]})"
                            break
                    else:
                        label = ""
                self.grid[y][x]['text'] = label


class Player:
    def __init__(self, ID):
        self.id = ID


if __name__ == '__main__':

    num_players = 2

    match = MatchGUI(
        players=[
            Player('Human'),
            Player("NotHuman"),
        ]
    )