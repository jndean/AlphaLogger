import numpy as np

import logger


state = logger.LoggerState(2)

x = state.toarray()
x[0, 0, 0] = 100
del x

print(state.toarray()[0])