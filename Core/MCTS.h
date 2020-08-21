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


typedef struct MCTS_{
	MCTSNode* root_node;
	MCTSNode* current_leaf_node;
} MCTS;


MCTS* MCTS_new();
void MCTS_reset(MCTS* mcts, uint8_t num_players);
void MCTS_free(MCTS* mcts);
void MCTS_reset_with_positions(MCTS* mcts, uint8_t num_players, Vec2* positions);
void MCTS_search_part1(MCTS* mcts, int8_t* inference_array);


#endif  /* MCTS_H */