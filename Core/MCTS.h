#ifndef MCTS_H
#define MCTS_H

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <math.h>

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
    uint32_t N[5 * 5 * 10];
    float W[5 * 5 * 10];

    uint32_t sumN;
    int current_move_idx;

} MCTSNode;


typedef struct MCTS_{
	MCTSNode* root_node;
	MCTSNode* current_leaf_node;
} MCTS;


MCTS* MCTS_new();
void MCTS_init(MCTS* mcts);
void MCTS_free(MCTS* mcts);
void MCTSNode_unpack_inference(MCTSNode* node, float* P, float* V);
void MCTS_init_with_state(MCTS* mcts, LoggerState* state);
void MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array);
void MCTS_search_backward_pass(MCTS* mcts);
int MCTS_choose_move_greedy(MCTS* mcts);


#endif  /* MCTS_H */