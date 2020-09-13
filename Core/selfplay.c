
#include "selfplay.h"


// --------------------------------------- Parallelised self-play ----------------------------------- //


/*
 TODO:
   + Data augmentations
      - Random choose to keep samples, making them sparser
      - Transform data through board symmetries
   + Stop OMP threads rejoining, make thread 0 do the single-threaded parts
*/

void self_play(PyObject* inference_method, int num_samples) {

/*
  //omp_set_num_threads(10);
  const int batch_size = 1;

  // Create numpy arrays for outputs
  npy_intp states_dims[] = {num_samples, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  npy_intp probs_dims[]  = {num_samples, 5, 5, 10};
  npy_intp scores_dims[] = {num_samples, NUM_PLAYERS};
  PyObject* states_arr = PyArray_SimpleNew(4, states_dims , NPY_INT8);
  PyObject* probs_arr  = PyArray_SimpleNew(4, probs_dims  , NPY_FLOAT32);
  PyObject* scores_arr = PyArray_SimpleNew(2, scores_dims, NPY_FLOAT32);
  MALLOC_CHECK(states_arr);
  MALLOC_CHECK(probs_arr);
  MALLOC_CHECK(scores_arr);
  int8_t* states_data = PyArray_GETPTR1((PyArrayObject*) states_arr, 0);
  int8_t* probs_data  = PyArray_GETPTR1((PyArrayObject*) probs_arr , 0);
  int8_t* scores_data = PyArray_GETPTR1((PyArrayObject*) scores_arr, 0);


  // Create an inference wrapper around part of the 'states_arr' memory
  npy_intp inference_dims[] = {batch_size, 5, 5, NUM_STATE_ARRAY_CHANNELS};
  PyObject* inference_arr = PyArray_SimpleNewFromData(4, inference_dims, NPY_INT8, states_data);
  MALLOC_CHECK(inference_arr);

  const int input_stride = NUM_SQUARES * NUM_STATE_ARRAY_CHANNELS;

  // Set up MCTS managers
  MCTS* mcts_array[batch_size];
  for (int i = 0; i < batch_size; ++i) {
    MCTS* mcts = MCTS_new(inference_method);
    // MCTS_init(mcts, NULL);  // Manual init, so inference can be batched
    LoggerState_getstatearray(&mcts->root_node->state, &input_data[i * input_stride]);
    mcts_array[i] = mcts;
  }
*/
  //   // Perform batched inference on the root game states
  //   PyObject* inference_args = PyTuple_Pack(1, input_arr);
  //   PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
  //   float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
  //   float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);
  //   for (int i = 0; i < batch_size; ++i) {
  //     MCTSNode_unpack_inference(mcts_array[i]->root_node, &P[i * NUM_MOVES], &V[i * NUM_PLAYERS]);
  //   }
  //   Py_DECREF(P_and_V);

  // // Main play loop
  // for (int move_num = 0; move_num < 1; ++move_num) {

  //   // Conduct num_simulations searches
  //   for (int s = 0; s < num_simulations; ++s) {

  //     // #pragma omp parallel for
  //     for (int i = 0; i < batch_size; ++i) {
  //       MCTS_search_forward_pass(mcts_array[i], &input_data[input_stride * i]);
  //     }

  //     // Perform batched inference on the leaf game states
  //     P_and_V = PyObject_CallObject(inference_method, inference_args);
  //     P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
  //     V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);

  //     // Unpack infernces into leaf nodes and backpropogate scores
  //     for (int i = 0; i < batch_size; ++i) {
  //       MCTS* mcts = mcts_array[i];

  //       MCTSNode_unpack_inference(mcts->current_leaf_node, &P[i * NUM_MOVES], &V[i * NUM_PLAYERS]);
        
  //       MCTS_search_backward_pass(mcts);
  //     }

  //     Py_DECREF(P_and_V);
  //   }
  // }


  // Py_DECREF(input_arr);
  // Py_DECREF(inference_args);
  // for(int i = 0; i < batch_size; ++i) {
  //   MCTS_free(mcts_array[i]);
  // }
}