CC=gcc

PYFLAGS=-I/usr/include/python3.8/ -I/home/josef/AlphaLogger/venv/lib/python3.8/site-packages/numpy/core/include/
OMPFLAGS=-lgomp -fopenmp
CFLAGS=-fPIC -Wall -Wno-unused-function -g


default: logger.o MCTS.o selfplay.o core.o
	$(CC) $(OMPFLAGS) $(CFLAGS) -shared $^ -o core.so

core.o: core.c
	$(CC) $(PYFLAGS) $(OMPFLAGS) $(CFLAGS) -c $^ -o $@

%.o: %.c
	$(CC) $(OMPFLAGS) $(PYFLAGS) $(CFLAGS) -c -o $@ $^

clean:
	rm logger.o MCTS.o core.o