# AlphaLogger

Implementing the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm (the entirely unsupervised iteration of AlphaGo), and then training it to play the beautiful game of Logger. The Logger state-space is a *little* smaller than that of Go or Chess, but then my GPU is a *little* smaller than a rack of 5,000 TPUs.

Unlike AlphaZero, AlphaLogger supports more than 2 player, by ending the network value head with a softmax across a vector of player win probabilities which are later mapped to [-1, 1].



Proof-of-concept tasks:

- [x] A python implementation of the Logger game
- [x] A GUI to visualise agent behaviour and let a Human play against the machine
- [x] A python implementation of the Monte Carlo Tree Search
- [x] A python implementation of the self-play and training pipeline

- [x] A Keras implementation of the AlphaZero ResNet, and an agent applying it via the MCTS

The proof-of-concept is now complete, but it is slow beyond the point of viability.

High-speed version tasks (using the CPython C API):

- [x] An efficient and thread-safe C implementation of Logger 
- [x] An efficient and thread-safe C implementation of the Monte Carlo Tree Search
- [x] A fully parallelised self-play and training pipeline supporting batching inference over simultaneous games

Done! This was more work than I intended for a quick project...

The high-speed implementation makes decent use of all available hardware, maxing out any number of CPU threads and achieving ~70% GPU compute utilisation if the neural network is big enough. In an isolated test using 8 threads on a mobile i5-8250U (not the training machine) it can make >20 million Logger moves per second or run ~1.5 million forward and backward passes per second on the MCTS (when using random state value estimations).

I trained a model for ~15 hrs, performing self-play for 50,000 moves across 128 simultaneous game boards each epoch to generate updated training data. Each self-play epoch was followed by two retraining epochs on the generated data. The resultant model was kinda trash, but did at least defeat another agent using an identical Monte Carlo Tree Search to enhance a flat probability distribution.

Possible improvements:

- Commit to training for a long time
- Data augmentation using the 8 symmetries of the Logger board
- Devise a board representation that allows the spatial placement of 'protester' pieces to be represented in some spatial dimension of the network (the same as 'player' placement), rather than just as indexes.