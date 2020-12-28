
#include "MCTS.h"


/* 
   Leave 'parent_node' as NULL to imply this is the root node
   Leave 'state' as NULL to start with a random starting state
*/
void MCTSNode_init(MCTSNode* node, MCTSNode* parent_node, LoggerState* state) {
    memset(node->children, 0, sizeof(node->children));
    memset(node->N, 0, sizeof(node->N));
    memset(node->W, 0, sizeof(node->W));
    node->parent = parent_node;
    node->sumN = 0;

    if (state != NULL) {
        memcpy(&node->state, state, sizeof(LoggerState));        
    } else {
        LoggerState_reset(&node->state);
    }
}


void MCTSNode_free(MCTSNode* node) {
    for (int i=0; i<NUM_MOVES; ++i) {
        if (node->children[i] != NULL) {
            MCTSNode_free(node->children[i]);
        }
    }
    free(node);
}


void MCTSNode_unpack_inference(MCTSNode* node, float* P, float* V) {
    memcpy(node->P, P, sizeof(node->P));
    // The infered V always has the current player first, (so the NN understands the current player)
    // whereas the MCTSNode stores things with player 0 first. Convert during the copy.
    const int current_player = node->state.current_player;
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        node->V[(i + current_player) % NUM_PLAYERS] = V[i];
    }
}


void MCTSNode_compute_mcts_probs(MCTSNode* node, float* out) {
    uint32_t* N = node->N;
    float over_sum_N = 1 / node->sumN;
    for (int i = 0; i < NUM_MOVES; ++i) {
        *(out++) = *(N++) * over_sum_N;
    }
}


MCTSNode* MCTSNode_create_child(MCTSNode* node, int move_idx) {
    MCTSNode* child_node = malloc(sizeof(MCTSNode));
    MALLOC_CHECK(child_node);
    node->children[move_idx] = child_node;
    MCTSNode_init(child_node, node, &node->state);

    Move move = {
        .y = move_idx / (5 * 10),
        .x = (move_idx / 10) % 5, 
        .action = move_idx % 10, 
        .protest_y = 0, 
        .protest_x = 0
    };
    LoggerState_domove(&child_node->state, move);

    return child_node;
}


void MCTSNode_init_as_root(MCTSNode* node, LoggerState* state, PyObject* inference_method) {
    MCTSNode_init(node, NULL, state);

    npy_intp input_dims[] = {1, 5, 5, NUM_STATE_ARRAY_CHANNELS};
    PyObject* input_arr = PyArray_SimpleNew(4, input_dims, NPY_INT8);
    MALLOC_CHECK(input_arr);
    int8_t* input_data = PyArray_GETPTR1((PyArrayObject*) input_arr, 0);
    PyObject* inference_args = PyTuple_Pack(1, input_arr);

    // Put the data in place
    LoggerState_getstatearray(&node->state, input_data);

    // Do inference
    PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
    float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
    float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);

    // Copy out the outputs
    MCTSNode_unpack_inference(node, P, V);

    Py_DECREF(P_and_V);
    Py_DECREF(input_arr);
    Py_DECREF(inference_args);
}

// -------------------------------- The MCTS container object --------------------------- //


MCTS* MCTS_new(PyObject* inference_method) {
    MCTS* mcts = malloc(sizeof(MCTS));
    MALLOC_CHECK(mcts);
    mcts->inference_method = inference_method;
    Py_INCREF(inference_method);
    mcts->root_node = NULL;
    mcts->current_leaf_node = NULL;
    return mcts;
}

void MCTS_free(MCTS* mcts) {
    if (mcts->root_node != NULL)
        MCTSNode_free(mcts->root_node);
    Py_DECREF(mcts->inference_method);
    free(mcts);
}


/* 
   Leave 'state' as NULL to start with a random starting state
*/
void MCTS_init(MCTS* mcts, LoggerState* state) {
    if (mcts->root_node != NULL) MCTSNode_free(mcts->root_node);
    mcts->root_node = NULL;
    mcts->current_leaf_node = NULL;

    if (state != NULL) {
        mcts->root_node = malloc(sizeof(MCTSNode));
        MALLOC_CHECK(mcts->root_node);
        MCTSNode_init_as_root(mcts->root_node, state, mcts->inference_method);
    }
}


/*
    Returns a boolean indicating if NN inference is required. E.g. infernce is not 
    required if the leaf node is a game-over state, since it's value is predefined
*/
int MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array) {
    // Special case: if there's no root node, create a random one, don't pick a move,
    // arrange the initial state for batched inference
    if (mcts->root_node == NULL) {
        mcts->root_node = malloc(sizeof(MCTSNode));
        MALLOC_CHECK(mcts->root_node);
        MCTSNode_init(mcts->root_node, NULL, NULL);
        mcts->current_leaf_node = mcts->root_node;
        LoggerState_getstatearray(&mcts->root_node->state, inference_array);
        return 1;
    }


    MCTSNode* node = mcts->root_node;
    int move_idx = -1;

    // Stochastically choose branches until a leaf node is reached
    while (1) {
        if (node->state.game_winner != -1) {
            mcts->current_leaf_node = node;
            return 0;
        }

        // Find the move maximising U
        float maxU = -FLT_MAX;
        float sqrt_sumN = sqrt((float)node->sumN);
        for (size_t i = 0; i < NUM_MOVES; ++i) {
            if (!node->state.legal_moves[i])
                continue;
            int32_t N = node->N[i];
            float U = C_PUCT * node->P[i] * sqrt_sumN / (1 + N);
            if (N != 0) {
                U += node->W[i] / N;
            }
            if (U > maxU) {
                move_idx = i;
                maxU = U;
            }
        }

        // Record what move was made for backpropogation later
        node->current_move_idx = move_idx;

        // Move down the branch
        MCTSNode* next_node = node->children[move_idx];
        if (next_node == NULL)
            break;
        node = next_node;
    }
    
    // Create the new leaf node
    MCTSNode* leaf_node = MCTSNode_create_child(node, move_idx);
    mcts->current_leaf_node = leaf_node;

    // No inference required if there's a winner
    if (leaf_node->state.game_winner != -1) {
        for (int i = 0; i < NUM_PLAYERS; ++i)
            leaf_node->V[i] = -1;
        leaf_node->V[leaf_node->state.game_winner] = 1;
        return 0;
    }

    // Copy the game state into the inference batch for the NN
    LoggerState_getstatearray(&leaf_node->state, inference_array);
    return 1;
}


void MCTS_search_backward_pass(MCTS* mcts) {

    float* V = mcts->current_leaf_node->V;
    MCTSNode* node = mcts->current_leaf_node;

    while (NULL != (node = node->parent)) {
        int move_idx = node->current_move_idx;
        int current_player = node->state.current_player;

        node->W[move_idx] += V[current_player];
        node->N[move_idx]++;
        node->sumN++;
    }

}


int MCTS_choose_move_greedy(MCTS* mcts) {
    uint32_t* N = mcts->root_node->N;
    int8_t* legal_moves = mcts->root_node->state.legal_moves;
    int argmax = -1;
    uint32_t max = 0;
    for (int i = 0; i < NUM_MOVES; ++i) {
        if (legal_moves[i] && N[i] > max) {
            max = N[i];
            argmax = i;
        }
    }
    return argmax;
}


int MCTS_choose_move_exploratory(MCTS* mcts) {
    uint32_t* N = mcts->root_node->N;
    int8_t* legal_moves = mcts->root_node->state.legal_moves;

    // TODO: add Dirichlet noise

    uint32_t sumN = mcts->root_node->sumN;
    uint32_t choice = ((uint32_t) rand()) % sumN;
    uint32_t partial_sum = 0;
    for (int i = 0; i < NUM_MOVES; ++i) {
        if (legal_moves[i]) {
            partial_sum += N[i];
            if (partial_sum > choice)
                return i;
        } 
    }

    printf("This shouldn't happen\n");
    exit(1702);
}


void MCTS_run_simulations(MCTS* mcts, int num_simulations) {

    // Create numpy arrays for inference
    static npy_intp input_dims[] = {1, 5, 5, NUM_STATE_ARRAY_CHANNELS};
    PyObject* input_arr = PyArray_SimpleNew(4, input_dims, NPY_INT8);
    MALLOC_CHECK(input_arr);
    int8_t* input_data = PyArray_GETPTR1((PyArrayObject*) input_arr, 0);
    PyObject* inference_args = PyTuple_Pack(1, input_arr);

    for (int i = 0; i < num_simulations; ++i) {
        // Do a forward pass, creating a leaf node and putting it's state in input_data
        int inference_required = MCTS_search_forward_pass(mcts, input_data);

        if (inference_required) {
            // Perform inference to compute the P and V for the leaf node
            PyObject* P_and_V = PyObject_CallObject(mcts->inference_method, inference_args);
            float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
            float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);
            MCTSNode_unpack_inference(mcts->current_leaf_node, P, V);
            Py_DECREF(P_and_V);
        }

        // Backward pass, propagating the visit counts and V values back up the tree
         MCTS_search_backward_pass(mcts);
    }

    Py_DECREF(input_arr);
    Py_DECREF(inference_args);
}


void MCTS_do_move(MCTS* mcts, int move_idx)
{
    MCTSNode* old_root = mcts->root_node;

    // Make the corresponding child node the new root node
    // Create a node if no such child exists
    mcts->root_node = old_root->children[move_idx];
    if (mcts->root_node == NULL) {
        mcts->root_node = MCTSNode_create_child(old_root, move_idx);
        printf("Child with no inference: %d\n", move_idx);
        // printf("This creates a child that has had no inference performed?\n");
        // printf("Put 'inferred' flag in node, use it for root_node too?\n");
    }
    mcts->root_node->parent = NULL;
    mcts->current_leaf_node = NULL;

    old_root->children[move_idx] = NULL;
    MCTSNode_free(old_root);
}

