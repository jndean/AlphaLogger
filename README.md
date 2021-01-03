# AlphaLogger

Implementing the AlphaZero algorithm (the entirely unsupervised iteration of AlphaGo) to learn the beautiful game of Logger. The Logger state-space is a *little* smaller than that of Go or Chess, but then my GPU is a *little* smaller than a V100.



Proof-of-concept tasks:

- [x] A python implementation of the Logger game
- [x] A GUI to visualise agent behaviour and let a Human play against the machine
- [x] A python implementation of the Monte Carlo Tree Search
- [x] A python implementation of the self-play and training pipeline

- [x] A Keras implementation of the AlphaZero ResNet, and an agent applying it via the MCTS

The proof-of-concept is now complete, but it is sloooow beyond the point of viability.



Highspeed version Tasks:

- [x] An efficient and thread-safe C implementation of Logger 
- [x] An efficient and thread-safe C implementation of the MCTS
- [x] A fully parallelised self-play and training pipeline supporting batching inference over simultaneous games



The C implementation of Logger can handle >20 million moves per second using 8 threads on a mobile i5-8250U.
The C implementation of the Monte Carlo Tree Search can do ~1.5 million forward & backward passes of the on the tree per second using the same hardware. This is when using a constant inference function, not the neural network
