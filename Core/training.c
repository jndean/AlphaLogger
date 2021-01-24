
#include "training.h"



/*
 TODO:
   + Data augmentations
      - Random choose to keep samples, making them sparser
      - Transform data through board symmetries
*/


PyObject* self_play(PyObject* inference_method, int num_samples, int num_simulations) {

  const int batch_size = 128;

  int num_moves = num_samples / batch_size;
  num_samples = num_moves * batch_size;

  // Create numpy arrays
  npy_intp inference_dims[] = {batch_size, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  npy_intp states_dims[]    = {num_samples, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  npy_intp probs_dims[]     = {num_samples, 5 * 5 * 10};
  npy_intp scores_dims[]    = {num_samples, NUM_PLAYERS};
  PyObject* inference_arr = PyArray_SimpleNew(4, inference_dims, NPY_INT8);
  PyObject* states_arr    = PyArray_SimpleNew(4, states_dims   , NPY_INT8);
  PyObject* probs_arr     = PyArray_SimpleNew(2, probs_dims    , NPY_FLOAT32);
  PyObject* scores_arr    = PyArray_SimpleNew(2, scores_dims   , NPY_FLOAT32);
  MALLOC_CHECK(inference_arr);
  MALLOC_CHECK(states_arr);
  MALLOC_CHECK(probs_arr);
  MALLOC_CHECK(scores_arr);
  int8_t* states_data    = PyArray_GETPTR1((PyArrayObject*) states_arr   , 0);
  float*  probs_data     = PyArray_GETPTR1((PyArrayObject*) probs_arr    , 0);
  float*  scores_data    = PyArray_GETPTR1((PyArrayObject*) scores_arr   , 0);


  for (int i = 0; i < num_samples * NUM_PLAYERS; ++i) {
    scores_data[i] = -1;
  }

  // Set up MCTS managers
  MCTS* mcts_array[batch_size];
  uint16_t move_counts[batch_size];
  for (int thread = 0; thread < batch_size; ++thread) {
    MCTS* mcts = MCTS_new(inference_method);
    MCTS_init(mcts, NULL);
    mcts_array[thread] = mcts;
    move_counts[thread] = 0;
  }

  // Main selfplay loop
  for (int m = 0; m < num_moves; ++m) {

      MCTS_run_batched_simulations(mcts_array,
				  inference_method,
				  batch_size,
				  num_simulations,
				   inference_arr);
      
      #pragma omp parallel for
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
      
      printf("\rmove %d/%d", m+1, num_moves);
      fflush(stdout);
  }
  printf("\n");

  // Assign a draw to unfinished games
  for (int thread = 0; thread < batch_size; ++thread) {
    float* score = &scores_data[thread * NUM_PLAYERS];
    for (int i = 0; i < move_counts[thread]; ++i) {
      score -= NUM_PLAYERS * batch_size;
      for (int j = 0; j < NUM_PLAYERS; ++j) {
        score[j] = 0;
      }
    }
  }

  // Clear up working space
  Py_DECREF(inference_arr);
  for(int thread = 0; thread < batch_size; ++thread) {
    MCTS_free(mcts_array[thread]);
  }

  // Return tuple of arrays
  return PyTuple_Pack(3, states_arr, probs_arr, scores_arr);
}



void mak_opponent_move
