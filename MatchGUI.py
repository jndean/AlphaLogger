from collections import deque
import tkinter as tk
import tkinter.font as font

from Player import HumanPlayer, RandomPlayer


class MatchGUI:

    def __init__(self, players):
        self.players = deque(players)

        # Initialise GUI
        self.window = tk.Tk()
        grid_frame = tk.Frame()
        self.grid = [[], [], [], [], []]
        for row_num in range(5):
            for col_num in range(5):
                label = tk.Label(
                    width=10, height=5, borderwidth=2, relief="groove", master=grid_frame
                )
                label.grid(row=row_num, column=col_num)
                self.grid[row_num].append(label)
        grid_frame.pack()

        button_frame = tk.Frame()
        for i, ((y, x), label) in enumerate([
            ((0, 2), "↟"),
            ((1, 1), "↖"), ((1, 2), "↑"), ((1, 3), "↗"),
            ((2, 0), "↞"), ((2, 1), "←"), ((2, 2), "."), ((2, 3), "→"), ((2, 4), "↠"),
            ((3, 1), "↙"), ((3, 2), "↓"), ((3, 3), "↘"),
            ((4, 2), "↡")
        ]):
            button = tk.Button(text=label, width=3, height=1, master=button_frame)
            button['font'] = tk.font.Font(size=20)
            button.grid(row=y, column=x)
        button_frame.pack()
        self.grid[0][1]['text'] = "P2"
        self.window.mainloop()


if __name__ == '__main__':
    match = MatchGUI(
        players=[
            HumanPlayer('Human'),
            RandomPlayer('Random')
        ]
    )