
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


  // Set up MCTS managers
  MCTS* mcts_array[batch_size];
  float* game_starts[batch_size];
  for (int i = 0; i < batch_size; ++i) {
    MCTS* mcts = MCTS_new(inference_method);
    MCTS_init(mcts, NULL);
    mcts_array[i] = mcts;
    game_starts[i] = &scores_data[i * NUM_PLAYERS];
  }

  // Main play loop
  for (int m = 0; m < num_moves; ++m) {

    // Conduct num_simulations searches
    for (int s = 0; s < num_simulations; ++s) {

      #pragma omp parallel for
      for (int i = 0; i < batch_size; ++i) {
        MCTS_search_forward_pass(mcts_array[i], &inference_data[i * NUM_STATE_ARRAY_ELEMENTS]);
      }

      // Perform batched inference on the leaf game states
      PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
      float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
      float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);

      // Unpack infernces into leaf nodes and backpropogate scores
      #pragma omp parallel for
      for (int i = 0; i < batch_size; ++i) {
        MCTSNode_unpack_inference(mcts_array[i]->current_leaf_node, &P[i * NUM_MOVES], &V[i * NUM_PLAYERS]);
        MCTS_search_backward_pass(mcts_array[i]);
      }

      Py_DECREF(P_and_V);
    }

  }

  // Clear up working space
  Py_DECREF(inference_arr);
  Py_DECREF(inference_args);
  for(int i = 0; i < batch_size; ++i) {
    MCTS_free(mcts_array[i]);
  }


  // Return tuple of arrays
  return PyTuple_Pack(1, states_arr, probs_arr, scores_arr);
}