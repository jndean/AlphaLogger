# AlphaLogger

Implementing the AlphaZero algorithm (the entirely unsupervised iteration of AlphaGo) to learn the beautiful game of Logger. The Logger state-space is a *little* smaller than that of Go or Chess, but then my GPU is a *little* smaller than a V100.

Unlike AlphaZero, AlphaLogger supports more than 2 player, but ending the network value head with a softmax across a vector of player win probabilities which can then be mapped to [-1, 1].



Proof-of-concept tasks:

- [x] A python implementation of the Logger game
- [x] A GUI to visualise agent behaviour and let a Human play against the machine
- [x] A python implementation of the Monte Carlo Tree Search
- [x] A python implementation of the self-play and training pipeline

- [x] A Keras implementation of the AlphaZero ResNet, and an agent applying it via the MCTS

The proof-of-concept is now complete, but it is sloooow beyond the point of viability.

High-speed version Tasks (using the CPython C API):

- [x] An efficient and thread-safe C implementation of Logger 
- [x] An efficient and thread-safe C implementation of the Monte Carlo Tree Search
- [x] A fully parallelised self-play and training pipeline supporting batching inference over simultaneous games



The high-speed implementation makes decent use of all available hardware, maxing out any number of CPU threads and achieving ~70% GPU compute utilisation if the neural network is big enough. In an isolated test using 8 threads on a mobile i5-8250U (not the training machine) it can make >20 million Logger moves per second or  run ~1.5 million forward and backward passes per second on the MCTS (when using random value estimations).

I trained a model for ~15 hrs, performing self-play for 50,000 moves across 128 simultaneous game boards each epoch to generate updated training data, then letting the network back-propagate each sample twice. The resultant model was kinda trash, but did at least defeat another agent using an identical MCTS to enhance a flat probability distribution.