from time import time

import numpy as np

import core
import player



S, P, V = core.self_play(player.uniform_inference, 10000, 50)

print(S[:, 0, 0, 5::10].flatten())
print(S.shape, P.shape, V.shape)
