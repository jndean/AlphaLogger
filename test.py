from time import time

import numpy as np

import logger


state = logger.LoggerState(2)


s = time()
state.test()
t = time() - s

print('Games per second:', 1000000 // t)