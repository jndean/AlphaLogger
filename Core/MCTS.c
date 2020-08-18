
#include "MCTS.h"

/*
    Doesn't the caller is responsible for initialising the LoggerState member
*/
void MCTSNode_init(MCTSNode* node) {
    memset(node->children, 0, sizeof(node->children));
    memset(node->N, 0, sizeof(node->N));
    memset(node->W, 0, sizeof(node->W));
    node->sumN = 0;
    node->sqrt_sumN = 0;
    printf("sizeof(node.children)=%ld", sizeof(node->children));
}

void MCTSNode_init_root(MCTSNode* node, uint8_t num_players) {
    MCTSNode_init(node);
    LoggerState_reset(&node->state, num_players);
}


MCTSNode* MCTSNode_search_part1(MCTSNode* root_node, int8_t* inference_array) {
    
    MCTSNode* node = root_node;
    int move_idx;  

    // Stochastically choose branches until a leaf node is reached
    while (1) {
        if (node->state.game_over) {
            printf("TODO: gameover\n");
            return NULL;
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

        MCTSNode* next_node = node->children[move_idx];
        if (next_node == NULL)
            break;
        node = next_node;
    }
    
    
    // Create the new leaf node
    MCTSNode* new_node = malloc(sizeof(MCTSNode));
    MCTSNode_init(new_node);
    memcpy(&new_node->state, &node->state, sizeof(node->state));
    Move move = {
        .y = move_idx / (5 * 10),
        .x = (move_idx / 10) % 5, 
        .action = move_idx % 10, 
        .protest_y = 0, 
        .protest_x = 0
    };
    LoggerState_domove(&new_node->state, move);
    

    // Copy the game state into the inference batch for the NN
    LoggerState_getstatearray(&new_node->state, inference_array);

    return new_node;
}
