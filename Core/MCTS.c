
#include "MCTS.h"


/*
    The caller is responsible for initialising the LoggerState member
*/
void MCTSNode_reset(MCTSNode* node, MCTSNode* parent_node) {
    memset(node->children, 0, sizeof(node->children));
    memset(node->N, 0, sizeof(node->N));
    memset(node->W, 0, sizeof(node->W));
    node->parent = parent_node;
    node->sumN = 0;
    node->sqrt_sumN = 0;
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


void MCTS_reset(MCTS* mcts, uint8_t num_players) {
    MCTSNode_reset(mcts->root_node, NULL);
    LoggerState_reset(&mcts->root_node->state, num_players);
    mcts->current_leaf_node = NULL;
}

void MCTS_reset_with_positions(MCTS* mcts, uint8_t num_players, Vec2* positions) {
    MCTS_reset(mcts, num_players);
    LoggerState_setpositions(&mcts->root_node->state, positions);
}


void MCTS_search_forward_pass(MCTS* mcts, int8_t* inference_array) {
    
    MCTSNode* node = mcts->root_node;
    int move_idx;  

    // Stochastically choose branches until a leaf node is reached
    while (1) {
        if (node->state.game_over) {
            printf("TODO: gameover\n");
            mcts->current_leaf_node = NULL;
            return;
        }

        // Find the move maximising U
        double maxU = -1;
        double sqrt_sumN = node->sqrt_sumN;
        for (size_t i = 0; i < 5*5*10; ++i) {
            if (!node->state.legal_moves[i])
                continue;
            int32_t N = node->N[i];
            double U = C_PUCT * node->P[i] * sqrt_sumN / (1 + N);
            if (N != 0) {
                U += node->W[i] / N;
            }
            if (U > maxU) {
                move_idx = i;
                maxU = U;
            }
        }

        printf("move_idx=%d\n", move_idx);

        MCTSNode* next_node = node->children[move_idx];
        if (next_node == NULL)
            break;
        node = next_node;
    }
    
    
    // Create the new leaf node
    MCTSNode* leaf_node = malloc(sizeof(MCTSNode));
    MALLOC_CHECK(leaf_node);
    node->children[move_idx] = leaf_node;
    MCTSNode_reset(leaf_node, node);
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
    //LoggerState_getstatearray(&leaf_node->state, inference_array);
    mcts->current_leaf_node = leaf_node;
    
}

