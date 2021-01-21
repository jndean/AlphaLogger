CC=gcc

PYFLAGS=-I/usr/include/python3.8/ -I/home/josef/Projects/AlphaLogger/venv/lib/python3.8/site-packages/numpy/core/include/
GSLFLAGS=-L/usr/lib/x86_64-linux-gnu/ -lgsl -lgslcblas -lm
OMPFLAGS=-lgomp -fopenmp
CFLAGS=-fPIC -Wall -Wno-unused-function -g


default: logger.o MCTS.o selfplay.o core.o
	$(CC) $^ -o core.so $(PYFLAGS) $(GSLFLAGS) $(OMPFLAGS) $(CFLAGS) -shared 

%.o: %.c
	$(CC) -c -o $@ $^ $(PYFLAGS) $(GSLFLAGS) $(OMPFLAGS) $(CFLAGS) 

clean:
	rm logger.o MCTS.o core.o selfplay.o
