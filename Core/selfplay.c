
#include "selfplay.h"


// --------------------------------------- Parallelised self-play ----------------------------------- //


/*
 TODO:
   + Data augmentations
      - Random choose to keep samples, making them sparser
      - Transform data through board symmetries
   + Stop OMP threads rejoining, make thread 0 do the single-threaded parts
*/

PyObject* self_play(PyObject* inference_method, int num_samples, int num_simulations) {

  const int batch_size = 1;
  omp_set_num_threads(batch_size);

  int num_moves = num_samples / batch_size;
  num_samples = num_moves * batch_size;

  // Create numpy arrays
  npy_intp inference_dims[] = {batch_size, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  npy_intp states_dims[]    = {num_samples, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  npy_intp probs_dims[]     = {num_samples, 5, 5, 10};
  npy_intp scores_dims[]    = {num_samples, NUM_PLAYERS};
  PyObject* inference_arr = PyArray_SimpleNew(4, inference_dims, NPY_INT8);
  PyObject* states_arr    = PyArray_SimpleNew(4, states_dims   , NPY_INT8);
  PyObject* probs_arr     = PyArray_SimpleNew(4, probs_dims    , NPY_FLOAT32);
  PyObject* scores_arr    = PyArray_SimpleNew(2, scores_dims   , NPY_FLOAT32);
  MALLOC_CHECK(inference_arr);
  MALLOC_CHECK(states_arr);
  MALLOC_CHECK(probs_arr);
  MALLOC_CHECK(scores_arr);
  int8_t* inference_data = PyArray_GETPTR1((PyArrayObject*) inference_arr, 0);
  int8_t* states_data    = PyArray_GETPTR1((PyArrayObject*) states_arr   , 0);
  float*  probs_data     = PyArray_GETPTR1((PyArrayObject*) probs_arr    , 0);
  float*  scores_data    = PyArray_GETPTR1((PyArrayObject*) scores_arr   , 0);

  PyObject* inference_args = PyTuple_Pack(1, inference_arr);

  for (int i = 0; i < num_samples * NUM_PLAYERS; ++i) {
    scores_data[i] = -1;
  }


  // Set up MCTS managers
  MCTS* mcts_array[batch_size];
  uint16_t move_counts[batch_size];
  char inference_required[batch_size];
  for (int thread = 0; thread < batch_size; ++thread) {
    MCTS* mcts = MCTS_new(inference_method);
    MCTS_init(mcts, NULL);
    mcts_array[thread] = mcts;
    move_counts[thread] = 0;
  }

  // Main selfplay loop
  for (int m = 0; m < num_moves; ++m) {

    // To choose a move, do num_simulations searches
    for (int s = 0; s < num_simulations; ++s) {

      // #pragma omp parallel for
      for (int thread = 0; thread < batch_size; ++thread) {
        inference_required[thread] = MCTS_search_forward_pass(
        	mcts_array[thread], &inference_data[thread * NUM_STATE_ARRAY_ELEMENTS]);
      }

      // Perform batched inference on the leaf game states
      PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
      float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
      float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);

      // Unpack inferences into leaf nodes and backpropogate scores
      // #pragma omp parallel for
      for (int thread = 0; thread < batch_size; ++thread) {
        // Don't overwrite the V values if they're set by an actual win
        if(inference_required[thread]) {
          MCTSNode_unpack_inference(mcts_array[thread]->current_leaf_node, &P[thread * NUM_MOVES], &V[thread * NUM_PLAYERS]);
        }
        MCTS_search_backward_pass(mcts_array[thread]);
      }

      Py_DECREF(P_and_V);
    }

    for (int thread = 0; thread < batch_size; ++thread) {
	    // Copy the state and enhanced probabilities out of the root nodes
	    MCTS* mcts = mcts_array[thread];
    	LoggerState_getstatearray(&mcts->root_node->state, &states_data[thread * NUM_STATE_ARRAY_ELEMENTS]);
      MCTSNode_compute_mcts_probs(mcts->root_node, &probs_data[thread * NUM_MOVES]);

    	// Advance the games one move
    	int move = MCTS_choose_move_exploratory(mcts);
    	MCTS_do_move(mcts, move);
      move_counts[thread] += 1;

      int8_t winner = mcts->root_node->state.game_winner;
    	if (winner != -1) {
        float* score = &scores_data[thread * NUM_PLAYERS];
        for (int i = 0; i < move_counts[thread]; ++i) {
          winner = (winner + NUM_PLAYERS - 1) % NUM_PLAYERS;
          score[winner] = 1;
          score -= NUM_PLAYERS * batch_size;
        }
        move_counts[thread] = 0;
    		MCTS_init(mcts, NULL);
    	} 
    }

    states_data += NUM_STATE_ARRAY_ELEMENTS * batch_size;
    probs_data += NUM_MOVES * batch_size;
    scores_data += NUM_PLAYERS * batch_size;
  }

  // Assign a draw to unfinished games
  for (int thread = 0; thread < batch_size; ++thread) {
    float* score = &scores_data[thread * NUM_PLAYERS];
    for (int i = 0; i <= move_counts[thread]; ++i) {
      for (int j = 0; j < NUM_PLAYERS; ++j) {
        score[j] = 0;
      }
      score -= NUM_PLAYERS * batch_size;
    }
  }

  // Clear up working space
  Py_DECREF(inference_arr);
  Py_DECREF(inference_args);
  for(int thread = 0; thread < batch_size; ++thread) {
    MCTS_free(mcts_array[thread]);
  }


  // Return tuple of arrays
  return PyTuple_Pack(3, states_arr, probs_arr, scores_arr);
}