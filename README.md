# AlphaLogger

Implementing the AlphaZero algorithm (the entirely unsupervised iteration of AlphaGo) to learn the beautiful game of Logger. The Logger state-space is a *little* smaller than that of Go or Chess, but then my GPU is a *little* smaller than a V100.



Tasks:

- [x] A POC python implementation of the Logger game
- [x] A GUI to visualise agent behaviour and let a Human play against the machine
- [x] A POC python implementation of the Monte Carlo Tree Search
- [x] A Keras implementation of the AlphaZero ResNet, and an agent applying it via the MCTS
- [x] A POC python implementation of the self-play and training pipeline

- [ ] An efficient and thread-safe C implementation of Logger [ IN PROGRESS ]

- [ ] An efficient and thread-safe C implementation of the MCTS
- [ ] A fully parallelised self-play and training pipeline