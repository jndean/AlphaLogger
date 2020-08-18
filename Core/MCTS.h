#ifndef MCTS_H
#define MCTS_H

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#include "logger.h"

#define C_PUCT 3

typedef struct MCTSNode_{
    LoggerState state;
    struct MCTSNode_* children[5 * 5 * 10];

    double P[5 * 5 * 10];
    double V[4];
    int32_t N[5 * 5 * 10];
    double W[5 * 5 * 10];

    int32_t sumN;
    double sqrt_sumN;

} MCTSNode;


#endif  /* MCTS_H */