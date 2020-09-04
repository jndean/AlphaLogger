#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<Python.h>
#include <numpy/arrayobject.h>

#include<omp.h>

#include "utils.h"
#include "logger.h"
#include "MCTS.h"


  
// ---------------------- Python LoggerState wrapper ----------------- //

typedef struct {
    PyObject_HEAD
    LoggerState* state;
} PyLoggerState;


static PyObject *
PyLoggerState_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

    PyLoggerState *self;
    self = (PyLoggerState *) type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    self->state = malloc(sizeof(LoggerState));
    if (self->state == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    return (PyObject *) self;
}


static void
PyLoggerState_dealloc(PyLoggerState *self)
{
  if (self->state != NULL) free(self->state);
  Py_TYPE(self)->tp_free((PyObject *) self);
}


static int
PyLoggerState_init(PyLoggerState *self, PyObject *args, PyObject *kwds)
{
  LoggerState_reset(self->state);
  // Vec2 positions[2] = {{1, 1}, {3, 3}};
  // LoggerState_setpositions(self->state, positions);
  return 0;
}


static PyObject*
PyLoggerState_getstatearray(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{
  const int num_channels = 4 + 3 * NUM_PLAYERS;
  npy_intp dims[] = {5, 5, num_channels};
  PyObject *out_arr = PyArray_SimpleNew(3, dims, NPY_INT8);
  if (out_arr == NULL) 
    return NULL;

  int8_t* out_data = PyArray_GETPTR1((PyArrayObject*) out_arr, 0);
  LoggerState_getstatearray(self->state, out_data);

  return out_arr;
}


static PyObject*
PyLoggerState_getlegalmovesarray(PyLoggerState *self, PyObject *Py_UNUSED(ignored))
{
  static npy_intp dims[] = {5, 5, 10};
  return PyArray_SimpleNewFromData(
    3, dims, NPY_INT8, self->state->legal_moves
  );
}


static PyObject*
PyLoggerState_getplayerpositions(PyLoggerState *self, PyObject *Py_UNUSED(ignored))
{
  LoggerState* state = self->state;
  PyObject* ret = PyTuple_New(NUM_PLAYERS);
  for (int i = 0; i < NUM_PLAYERS; ++i) {
    int p = (state->current_player + i) % NUM_PLAYERS;
    Vec2 coords = state->positions[p];
    PyObject* py_coords = Py_BuildValue("(ii)", coords.y, coords.x);
    PyTuple_SetItem(ret, i, py_coords);
  }
  return ret;
}


static PyObject*
PyLoggerState_domove(PyLoggerState *self, PyObject* args, PyObject* keywds)
{
  static char* kwlist[] = {"y", "x", "action", "protest_y", "protest_x", NULL};
  unsigned char y, x, action, protest_y=0, protest_x=0;

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "bbb|bb", kwlist,
                                     &y, &x, &action, &protest_y, &protest_x))
    return NULL;

  Move move = {y, x, action, protest_y, protest_x};
  LoggerState_domove(self->state, move);

  Py_RETURN_NONE;
}


static PyObject*
PyLoggerState_test(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{

  omp_set_num_threads(8);
  #pragma omp parallel for
  for (int game_num = 0; game_num < 1000000; ++game_num) {
    LoggerState* state = malloc(sizeof(LoggerState));
    LoggerState_reset(state);
    for (int move_num = 0; move_num < 25; ++move_num) {
        int move_idx = -1;
        for (int i = 0; i < sizeof(state->legal_moves); ++i) {
            if (state->legal_moves[i] && i % 10 != 8) {
                move_idx = i;
                break;
            }
        }
        Move move = {move_idx / 50, (move_idx / 10) % 5, move_idx % 10, 0, 0};
        LoggerState_domove(state, move);
    }
    free(state);
  }
  /*
  Move move = {.y = 3, .x = 2, .action = 4, .protest_y = 0, .protest_x = 0};
  LoggerState_domove(self->state, move);
  for (int i=0; i < 10; ++i){
    move = (Move){.y = 3, .x = 2, .action = 9, .protest_y = 0, .protest_x = 0};
    LoggerState_domove(self->state, move);
  }
  move = (Move){.y = 3, .x = 2, .action = 0, .protest_y = 0, .protest_x = 0};
  LoggerState_domove(self->state, move);
  */
  Py_RETURN_NONE;
}

static PyMethodDef PyLoggerState_methods[] = {
    {"get_state_array", (PyCFunction) PyLoggerState_getstatearray, METH_NOARGS,
     "Get the board state as a numpy array"},
    {"get_legal_moves_array", (PyCFunction) PyLoggerState_getlegalmovesarray, METH_NOARGS,
     "Get the legal move mask"},
    {"get_player_positions", (PyCFunction) PyLoggerState_getplayerpositions, METH_NOARGS,
     "Get the positions of the player, a tuple of tuples (y, x)"},
    {"do_move", (PyCFunction) PyLoggerState_domove, METH_VARARGS | METH_KEYWORDS,
     "Enact a move as the current player"},
    {"test", (PyCFunction) PyLoggerState_test, METH_NOARGS,
     "Testing method"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLoggerStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "logger.LoggerState",
    .tp_doc = "The state of a logger game",
    .tp_basicsize = sizeof(PyLoggerState),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyLoggerState_new,
    .tp_dealloc = (destructor) PyLoggerState_dealloc,
    .tp_init = (initproc) PyLoggerState_init,
    .tp_methods = PyLoggerState_methods,
};


// --------------------------- PyMCTS object -------------------------- //

typedef struct {
    PyObject_HEAD
    MCTS* mcts;
} PyMCTS;


static PyObject *
PyMCTS_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

    PyMCTS *self;
    self = (PyMCTS *) type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    self->mcts = MCTS_new();
    if (self->mcts == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    return (PyObject *) self;
}


static void
PyMCTS_dealloc(PyMCTS *self)
{
    if (self->mcts != NULL)  MCTS_free(self->mcts);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static int
PyMCTS_init(PyMCTS *self, PyObject *args, PyObject *kwds)
{
  MCTS_init(self->mcts);
  return 0;
}


static PyObject*
PyMCTS_sync_with_game(PyMCTS *self, PyObject *args)
{
  PyObject* game_state = NULL;
  if (!PyArg_ParseTuple(args, "O", &game_state))
    return NULL;
  if (Py_TYPE(game_state) != &PyLoggerStateType) {
    PyErr_SetString(PyExc_ValueError, "MCTS.sync_with_game requires LoggerState instance as an argument");
    return NULL;
  }

  MCTS_free(self->mcts);
  self->mcts = MCTS_new();
  MALLOC_CHECK(self->mcts);
  MCTS_init_with_state(self->mcts, ((PyLoggerState*)(game_state))->state);

  Py_RETURN_NONE;
}


static PyObject*
PyMCTS_choose_move(PyMCTS *self, PyObject *args, PyObject *kwargs)
{
  static char* kwlist[] = {"inferer", "num_simulations", "exploratory", NULL};
  PyObject* inferer = NULL;
  int num_simulations = 0, exploratory = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|p", kwlist, 
            &inferer, &num_simulations, &exploratory)) {
    return NULL;
  }

  int move = MCTS_choose_move(self->mcts, inferer, num_simulations, exploratory);
  if (move == -1) {
    Py_RETURN_NONE;
  }
  return PyLong_FromLong((long)move);
}


static PyObject*
PyMCTS_do_move(PyMCTS *self, PyObject *args)
{
  int move_idx = -1;
  if (!PyArg_ParseTuple(args, "i", &move_idx))
    return NULL;

  if (move_idx < 0 || move_idx >= 5 * 5 * 10) {
    PyErr_SetString(PyExc_ValueError, "Invalid move_idx given to MCTS.done_move");
    return NULL;
  }

  MCTS_do_move(self->mcts, move_idx);

  Py_RETURN_NONE;
}


static PyObject*
PyMCTS_test(PyMCTS *self, PyObject *Py_UNUSED(ignored))
{
  Vec2* positions = self->mcts->root_node->state.positions;
  for (int i = 0; i < NUM_PLAYERS; ++i) {
    printf("(%d, %d)\n", positions[i].y, positions[i].x);
  }

  Py_RETURN_NONE;
}

static PyMethodDef PyMCTS_methods[] = {
    // {"get_state_array", (PyCFunction) PyMCTS_getstatearray, METH_NOARGS,
    //  "Get the board state as a numpy array"},
    {"choose_move", (PyCFunction) PyMCTS_choose_move, METH_VARARGS | METH_KEYWORDS,
     "Assuming MCTS simulations have been run, pick a move to do"},
    {"done_move", (PyCFunction) PyMCTS_do_move, METH_VARARGS,
     "Tell the MCTS to update internal state due to a move being done"},
    {"sync_with_game", (PyCFunction) PyMCTS_sync_with_game, METH_VARARGS,
     "Initialise the root node with the given starting state"},
    {"test", (PyCFunction) PyMCTS_test, METH_NOARGS,
     "Testing method"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMCTSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "core.MCTS",
    .tp_doc = "An object managing a Monte Carlo Tree Search",
    .tp_basicsize = sizeof(PyMCTS),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyMCTS_new,
    .tp_dealloc = (destructor) PyMCTS_dealloc,
    .tp_init = (initproc) PyMCTS_init,
    .tp_methods = PyMCTS_methods,
};


// --------------------------- Methods -------------------------- //


static PyObject*
core_testMCTSsearch(PyObject* self, PyObject* args){
  
  // MCTS* mcts = MCTS_new();
  // Vec2 positions[2] = {{0, 0}, {1, 1}};
  // MCTS_reset_with_positions(mcts, positions);

  // MCTS_search_forward_pass(mcts, 0);

  // MCTS_free(mcts);
  Py_RETURN_NONE;
}


static PyObject*
core_testMCTSselfplay(PyObject* self, PyObject* args){

  PyObject* inference_method = NULL;
  if (!PyArg_ParseTuple(args, "O", &inference_method))
        return NULL;

  //omp_set_num_threads(10);
  const int batch_size = 1;
  const int num_simulations = 5;

  // Create numpy arrays for inference
  npy_intp input_dims[] = {batch_size, 5, 5, 4 + 3 * NUM_PLAYERS};
  PyObject* input_arr = PyArray_SimpleNew(4, input_dims, NPY_INT8);
  MALLOC_CHECK(input_arr);
  int8_t* input_data = PyArray_GETPTR1((PyArrayObject*) input_arr, 0);

  const int input_stride = 5 * 5 * (4 + 3 * NUM_PLAYERS);
  const int P_stride = 5 * 5 * 10;
  const int V_stride = NUM_PLAYERS;

  // Set up MCTS managers
  MCTS* mcts_array[batch_size];
  for (int i = 0; i < batch_size; ++i) {
    MCTS* mcts = MCTS_new();
    MCTS_init(mcts);
    LoggerState_getstatearray(&mcts->root_node->state, &input_data[i * input_stride]);
    mcts_array[i] = mcts;
  }

  // Perform batched inference on the root game states
  PyObject* inference_args = PyTuple_Pack(1, input_arr);
  PyObject* P_and_V = PyObject_CallObject(inference_method, inference_args);
  float* P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
  float* V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);
  for (int i = 0; i < batch_size; ++i) {
    MCTSNode_unpack_inference(mcts_array[i]->root_node, &P[i * P_stride], &V[i * V_stride]);
  }
  Py_DECREF(P_and_V);

  // Main play loop
  for (int move_num = 0; move_num < 1; ++move_num) {

    // Conduct num_simulations searches
    for (int s = 0; s < num_simulations; ++s) {

      // #pragma omp parallel for
      for (int i = 0; i < batch_size; ++i) {
        MCTS_search_forward_pass(mcts_array[i], &input_data[input_stride * i]);
      }

      // Perform batched inference on the leaf game states
      P_and_V = PyObject_CallObject(inference_method, inference_args);
      P = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 0), 0);
      V = PyArray_GETPTR1((PyArrayObject*) PyTuple_GET_ITEM(P_and_V, 1), 0);

      // Unpack infernces into leaf nodes and backpropogate scores
      for (int i = 0; i < batch_size; ++i) {
        MCTS* mcts = mcts_array[i];

        MCTSNode_unpack_inference(mcts->current_leaf_node, &P[i * P_stride], &V[i * V_stride]);
        
        MCTS_search_backward_pass(mcts);
      }

      Py_DECREF(P_and_V);
    }
  }


  Py_DECREF(input_arr);
  Py_DECREF(inference_args);
  for(int i = 0; i < batch_size; ++i) {
    MCTS_free(mcts_array[i]);
  }
  Py_RETURN_NONE;
}


// ------------------------------ Module Setup ------------------------------ //

static PyMethodDef CoreMethods[] = {
    {"test_MCTS_search",  core_testMCTSsearch, METH_VARARGS, ""},
    {"test_MCTS_selfplay",  core_testMCTSselfplay, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef coremodule = {
   PyModuleDef_HEAD_INIT,
   "core",  
   NULL, 
   -1,
   CoreMethods
};

 
PyMODINIT_FUNC PyInit_core(void)
{
  time_t t;
  srand((unsigned) time(&t));

  import_array();

    PyObject *m;
    if (PyType_Ready(&PyLoggerStateType) < 0)
        return NULL;
    if (PyType_Ready(&PyMCTSType) < 0)
        return NULL;

    m = PyModule_Create(&coremodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyLoggerStateType);
    if (PyModule_AddObject(m, "LoggerState", (PyObject *) &PyLoggerStateType) < 0) {
        Py_DECREF(&PyLoggerStateType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyMCTSType);
    if (PyModule_AddObject(m, "MCTS", (PyObject *) &PyMCTSType) < 0) {
        Py_DECREF(&PyMCTSType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}