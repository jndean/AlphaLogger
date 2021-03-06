
#include "MCTS.h"


/* 
  A bit of machinery to cache discarded MCTSNodes in the MCTS to save on mallocs
  Is thread-safe in this application because each MCTS is only touched by a single thread at a time.
*/
#ifdef CACHE_MCTS_NODES

static inline MCTSNode* MCTS_malloc_MCTSNode(MCTS* mcts) {
    MCTSNode* node;
    if (mcts->node_cache == NULL) {
        node = malloc(sizeof(MCTSNode));
        MALLOC_CHECK(node);
    } else {
        node = mcts->node_cache;
        mcts->node_cache = *((MCTSNode**) node);
    }
    return node;
}

static inline void MCTS_free_MCTSNode(MCTS* mcts, MCTSNode* node) {
    *((MCTSNode**) node) = mcts->node_cache;
    mcts->node_cache = node;
}

#else

static inline MCTSNode* MCTS_malloc_MCTSNode(MCTS* mcts) {
    MCTSNode* node = malloc(sizeof(MCTSNode));
    MALLOC_CHECK(node);
    return node;
}
static inline void MCTS_free_MCTSNode(MCTS* mcts, MCTSNode* node) {
    free(node);
}

#endif


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


void MCTSNode_uninit(MCTS* mcts, MCTSNode* node) {
    for (int i=0; i<NUM_MOVES; ++i) {
        if (node->children[i] != NULL) {
            MCTSNode_uninit(mcts, node->children[i]);
        }
    }
    MCTS_free_MCTSNode(mcts, node);
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


MCTSNode* MCTSNode_create_child(MCTSNode* node, int move_idx, MCTS* mcts) {
    MCTSNode* child_node = MCTS_malloc_MCTSNode(mcts);
    node->children[move_idx] = child_node;
    MCTSNode_init(child_node, node, &node->state);

    LoggerState_domove(&child_node->state, move_idx);

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

    mcts->rng = gsl_rng_alloc(gsl_rng_mt19937); // Mersenne Twister
    gsl_rng_set(mcts->rng, (unsigned long int) mcts);

    #ifdef CACHE_MCTS_NODES
    mcts->node_cache = NULL;
    #endif

    return mcts;
}

void MCTS_free(MCTS* mcts) {
    if (mcts->root_node != NULL)
        MCTSNode_uninit(mcts, mcts->root_node);
    Py_DECREF(mcts->inference_method);
    gsl_rng_free(mcts->rng);

    #ifdef CACHE_MCTS_NODES
    while (mcts->node_cache != NULL) {
        MCTSNode* node = mcts->node_cache;
        mcts->node_cache = *((MCTSNode**) node);
        free(node);
    }
    #endif

    free(mcts);
}



/* 
   Leave 'state' as NULL to start with a random starting state
*/
void MCTS_init(MCTS* mcts, LoggerState* state) {
    if (mcts->root_node != NULL) MCTSNode_uninit(mcts, mcts->root_node);
    mcts->root_node = NULL;
    mcts->current_leaf_node = NULL;

    if (state != NULL) {
        mcts->root_node = MCTS_malloc_MCTSNode(mcts);
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
        mcts->root_node = MCTS_malloc_MCTSNode(mcts);
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
    MCTSNode* leaf_node = MCTSNode_create_child(node, move_idx, mcts);
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


static double dirichlet_alphas[NUM_MOVES];
__attribute__((constructor)) 
void __initialize_dirichlet_alphas() {
    for (int i = 0; i < NUM_MOVES; ++i) dirichlet_alphas[i] = 10/26;
}

int MCTS_choose_move_exploratory(MCTS* mcts) {
    uint32_t* N = mcts->root_node->N;
    int8_t* legal_moves = mcts->root_node->state.legal_moves;

    // Generate Dirichlet noise according to the original paper
    double probs[NUM_MOVES];
    gsl_ran_dirichlet(mcts->rng, NUM_MOVES, dirichlet_alphas, probs);

    double sum_legal_moves = 0;
    for (int i = 0; i < NUM_MOVES; ++i) {
        if (legal_moves[i]) {
            double summand = N[i] + probs[i];
            probs[i] = summand;
            sum_legal_moves += summand;
        } else {
            probs[i] = 0;
        }
    }

    double choice = rand() * (sum_legal_moves / RAND_MAX);
    double partial_sum = 0;
    for (int i = 0; i < NUM_MOVES; ++i) {
        if (legal_moves[i]) {
            partial_sum += probs[i];
            if (partial_sum > choice)
                return i;
        } 
    }

    printf("This shouldn't happen\n");
    exit(1702);
}


void MCTS_run_batched_simulations(MCTS** mcts_array,
				  PyObject* inference_method,
				  int batch_size,
				  int num_simulations,
				  PyObject* np_inference_arr) {

    omp_set_num_threads(batch_size > MAX_THREADS ? MAX_THREADS : batch_size);

    int8_t* inference_data = PyArray_GETPTR1((PyArrayObject*) np_inference_arr, 0);
    PyObject* inference_args = PyTuple_Pack(1, np_inference_arr);
    
    char* inference_required = calloc(batch_size, sizeof(char));
    
    for (int s = 0; s < num_simulations; ++s) {
	
      // Do a forward pass, creating a leaf node and putting it's state in input_data
        #pragma omp parallel for
	for (int thread = 0; thread < batch_size; ++thread) {
	    inference_required[thread] = MCTS_search_forward_pass(
	        mcts_array[thread], &inference_data[thread * NUM_STATE_ARRAY_ELEMENTS]);
	}

	// Perform batched inference on the leaf game states
	PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
	float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
	float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);
	
	// Unpack inferences into leaf nodes and backpropogate scores
        #pragma omp parallel for
	for (int thread = 0; thread < batch_size; ++thread) {
	    // Don't overwrite the V values if they're set by an actual win
	    if(inference_required[thread]) {
		MCTSNode_unpack_inference(
	            mcts_array[thread]->current_leaf_node,
		    &P[thread * NUM_MOVES],
		    &V[thread * NUM_PLAYERS]
	        );
	    }
	    MCTS_search_backward_pass(mcts_array[thread]);
	}

	Py_DECREF(P_and_V);
    }

    //Py_DECREF(inference_arr);
    Py_DECREF(inference_args);
    free(inference_required);
}


void MCTS_run_batched_simulations_wrapper(MCTS** mcts_array,
					  int batch_size,
					  int num_simulations) {
    npy_intp np_inference_dims[] = {batch_size, 5, 5, NUM_STATE_ARRAY_CHANNELS};
    PyObject* np_inference_arr = PyArray_SimpleNew(4, np_inference_dims, NPY_INT8);
    MALLOC_CHECK(np_inference_arr);
    MCTS_run_batched_simulations(mcts_array,
				 mcts_array[0]->inference_method,
				 batch_size,
				 num_simulations,
				 np_inference_arr);
    Py_DECREF(np_inference_arr);
}


void MCTS_run_simulations(MCTS* mcts, int num_simulations) {
    MCTS_run_batched_simulations_wrapper(&mcts, 1, num_simulations);
}


void MCTS_do_move(MCTS* mcts, int move_idx)
{
    // Make the corresponding child node the new root node
    // Create a node if no such child exists
    MCTSNode* old_root = mcts->root_node;
    mcts->root_node = old_root->children[move_idx];

    if (mcts->root_node == NULL) {  
        mcts->root_node = MCTS_malloc_MCTSNode(mcts);
	LoggerState_domove(&old_root->state, move_idx);
	MCTSNode_init_as_root(mcts->root_node, &old_root->state, mcts->inference_method);
    }
    
    mcts->root_node->parent = NULL;
    mcts->current_leaf_node = NULL;

    old_root->children[move_idx] = NULL;
    MCTSNode_uninit(mcts, old_root);
}

