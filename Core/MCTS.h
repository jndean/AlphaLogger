#ifndef MCTS_H
#define MCTS_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ALPHALOGGER_PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <math.h>

#include<Python.h>
#include <numpy/arrayobject.h>

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
    PyObject* inference_method;
	MCTSNode* root_node;
	MCTSNode* current_leaf_node;
} MCTS;


MCTS* MCTS_new(PyObject* inference_method);
void MCTS_free(MCTS* mcts);
void MCTS_init(MCTS* mcts, LoggerState* state);
void MCTS_sync_with_game(MCTS* mcts, LoggerState* state);
void MCTSNode_unpack_inference(MCTSNode* node, float* P, float* V);
void MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array);
void MCTS_search_backward_pass(MCTS* mcts);
int MCTS_choose_move(MCTS* mcts, int num_simulations, int exploratory);
void MCTS_do_move(MCTS* mcts, int move_idx);


#endif  /* MCTS_H */