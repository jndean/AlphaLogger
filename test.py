from time import time

import core


core.test_method()


quit()

state = core.LoggerState(2)


s = time()
state.test()
t = time() - s

print('Games per second:', 1000000 // t)