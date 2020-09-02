
#include "MCTS.h"


/*
    The caller is responsible for initialising the LoggerState member
*/
void MCTSNode_init(MCTSNode* node, MCTSNode* parent_node) {
    memset(node->children, 0, sizeof(node->children));
    memset(node->N, 0, sizeof(node->N));
    memset(node->W, 0, sizeof(node->W));
    node->parent = parent_node;
    node->sumN = 0;
}

void MCTSNode_free(MCTSNode* node) {
    for (int i=0; i<5*5*10; ++i) {
        if (node->children[i] != NULL) {
            MCTSNode_free(node->children[i]);
        }
    }
    free(node);
}


// -------------------------------- The MCTS container object --------------------------- //


MCTS* MCTS_new() {
    MCTS* mcts = malloc(sizeof(MCTS));
    MALLOC_CHECK(mcts);
    mcts->root_node = malloc(sizeof(MCTSNode));
    MALLOC_CHECK(mcts->root_node);
    return mcts;
}

void MCTS_free(MCTS* mcts) {
    if (mcts->root_node != NULL)
        MCTSNode_free(mcts->root_node);
    free(mcts);
}


void MCTS_init(MCTS* mcts) {
    MCTSNode_init(mcts->root_node, NULL);
    LoggerState_reset(&mcts->root_node->state);
    mcts->current_leaf_node = NULL;
}


void MCTS_init_with_state(MCTS* mcts, LoggerState* state) {
    MCTS_init(mcts);
    memcpy(&mcts->root_node->state, state, sizeof(LoggerState));
}


void MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array) {
    
    MCTSNode* node = mcts->root_node;
    int move_idx = -1;

    // Stochastically choose branches until a leaf node is reached
    while (1) {
        if (node->state.game_winner != -1) {
            printf("TODO: gameover\n");
            mcts->current_leaf_node = NULL;
            return;
        }

        // Find the move maximising U
        float maxU = -1;
        float sqrt_sumN = sqrt((float)node->sumN);
        for (size_t i = 0; i < 5*5*10; ++i) {
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

        printf("fwd %p: move_idx=%d\n", node, move_idx);

        // Record what move was made for backpropogation later
        node->current_move_idx = move_idx;

        // Move down the branch
        MCTSNode* next_node = node->children[move_idx];
        if (next_node == NULL)
            break;
        node = next_node;
    }
    
    // Create the new leaf node
    MCTSNode* leaf_node = malloc(sizeof(MCTSNode));
    MALLOC_CHECK(leaf_node);
    node->children[move_idx] = leaf_node;
    MCTSNode_init(leaf_node, node);
    memcpy(&leaf_node->state, &node->state, sizeof(node->state));
    Move move = {
        .y = move_idx / (5 * 10),
        .x = (move_idx / 10) % 5, 
        .action = move_idx % 10, 
        .protest_y = 0, 
        .protest_x = 0
    };
    LoggerState_domove(&leaf_node->state, move);


    // Copy the game state into the inference batch for the NN
    LoggerState_getstatearray(&leaf_node->state, inference_array);
    mcts->current_leaf_node = leaf_node;
    
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


void MCTS_search_backward_pass(MCTS* mcts) {

    float* V = mcts->current_leaf_node->V;
    MCTSNode* node = mcts->current_leaf_node;

    while (NULL != (node = node->parent)) {
        int move_idx = node->current_move_idx;
        int current_player = node->state.current_player;

        node->W[move_idx] = V[current_player];
        node->N[move_idx]++;
        node->sumN++;

        printf("bkwd %p\n", node);
    }

}


int _choose_move_greedy(MCTS* mcts) {
    uint32_t* N = mcts->root_node->N;
    int8_t* legal_moves = mcts->root_node->state.legal_moves;
    int argmax = -1;
    uint32_t max = 0;
    for (int i = 0; i < 5 * 5 * 10; ++i) {
        if (legal_moves[i] && N[i] > max) {
            max = N[i];
            argmax = i;
        }
    }
    return argmax;
}


int _choose_move_exploratory(MCTS* mcts) {
    uint32_t* N = mcts->root_node->N;
    int8_t* legal_moves = mcts->root_node->state.legal_moves;

    uint8_t sumN = mcts->root_node->sumN;
    if (sumN == 0) return -1;
    uint32_t choice = ((uint32_t) rand()) % sumN;

    uint32_t partial_sum = 0;
    int i;
    for (i = 0; i < 5 * 5 * 10; ++i) {
        if (legal_moves[i]) {
            partial_sum += N[i];
            if (partial_sum > choice)
                break;
        } 
    }
    return i;
}


int MCTS_choose_move(MCTS* mcts, int exploratory) {
    if (exploratory) 
        return _choose_move_exploratory(mcts);
    return _choose_move_greedy(mcts);
}
