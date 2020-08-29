#ifndef MCTS_H
#define MCTS_H

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#include<Python.h>

#include "utils.h"
#include "logger.h"

#define C_PUCT 3

typedef struct MCTSNode_{
    LoggerState state;
    struct MCTSNode_* children[5 * 5 * 10];
    struct MCTSNode_* parent;

    float P[5 * 5 * 10];
    float V[NUM_PLAYERS];
    int32_t N[5 * 5 * 10];
    float W[5 * 5 * 10];

    int32_t sumN;
    float sqrt_sumN;

} MCTSNode;


typedef struct MCTS_{
	MCTSNode* root_node;
	MCTSNode* current_leaf_node;
} MCTS;


MCTS* MCTS_new();
void MCTS_reset(MCTS* mcts);
void MCTS_free(MCTS* mcts);
void MCTS_reset_with_positions(MCTS* mcts, Vec2* positions);
void MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array);
void MCTS_search_backward_pass(MCTS* mcts);


#endif  /* MCTS_H */