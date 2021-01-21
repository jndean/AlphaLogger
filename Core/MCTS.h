#ifndef MCTS_H
#define MCTS_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ALPHALOGGER_PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY

#define C_PUCT 4
#define MAX_THREADS 8
#define CACHE_MCTS_NODES 

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<float.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include<Python.h>
#include <numpy/arrayobject.h>

#include "utils.h"
#include "logger.h"


typedef struct MCTSNode_{
    LoggerState state;
    struct MCTSNode_* children[NUM_MOVES];
    struct MCTSNode_* parent;

    float P[NUM_MOVES];
    float V[NUM_PLAYERS];
    uint32_t N[NUM_MOVES];
    float W[NUM_MOVES];

    uint32_t sumN;
    int current_move_idx;

} MCTSNode;


typedef struct MCTS_{
    PyObject* inference_method;
    MCTSNode* root_node;
    MCTSNode* current_leaf_node;
    gsl_rng* rng;

#ifdef CACHE_MCTS_NODES
    MCTSNode* node_cache;
#endif
} MCTS;


MCTS* MCTS_new(PyObject* inference_method);
void MCTS_free(MCTS* mcts);
void MCTS_init(MCTS* mcts, LoggerState* state);
void MCTSNode_unpack_inference(MCTSNode* node, float* P, float* V);
void MCTSNode_compute_mcts_probs(MCTSNode* node, float* out);
int MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array);
void MCTS_search_backward_pass(MCTS* mcts);
void MCTS_run_simulations(MCTS* mcts, int num_simulations);
int MCTS_choose_move_exploratory(MCTS* mcts);
int MCTS_choose_move_greedy(MCTS* mcts);
void MCTS_do_move(MCTS* mcts, int move_idx);


#endif  /* MCTS_H */
