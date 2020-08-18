#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<Python.h>
#include <numpy/arrayobject.h>

#include<omp.h>

#include"logger.h"



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
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static int
PyLoggerState_init(PyLoggerState *self, PyObject *args, PyObject *kwds)
{
  PyObject *num_players = NULL;
    if(!PyArg_ParseTuple(args, "O", &num_players))
      return -1;

  LoggerState_reset(self->state, PyLong_AsLong(num_players));
  return 0;
}


static PyObject*
PyLoggerState_getstatearray(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{
  const int num_channels = 4 + 3 * self->state->num_players;
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
  PyObject* ret = PyTuple_New(state->num_players);
  for (int i = 0; i < state->num_players; ++i) {
    int p = (state->current_player + i) % state->num_players;
    Vec2 coords = state->positions[p];
    PyObject* py_coords = Py_BuildValue("(ii)", coords.y, coords.x);
    PyTuple_SetItem(ret, i, py_coords);
  }
  return ret;
}


static PyObject*
PyLoggerState_domove(PyLoggerState *self, PyObject* args, PyObject* keywds)
{
  static char* kwlist[] = {"y", "x", "action", "protest_y", "protest_x"};
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

  omp_set_num_threads(10);
  #pragma omp parallel for
  for (int game_num = 0; game_num < 1000000; ++game_num) {
    LoggerState* state = malloc(sizeof(LoggerState));
    LoggerState_reset(state, self->state->num_players);
    for (int move_num = 0; move_num < 25; ++move_num) {
        int move_idx;
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


// --------------------------- Methods -------------------------- //


PyObject* core_testmethod(PyObject* self, PyObject* args){
  
  Py_RETURN_NONE;
}

// ------------------------------ Module Setup ------------------------------ //

static PyMethodDef CoreMethods[] = {
    {"test_method",  core_testmethod, METH_VARARGS,
     "A test method."},
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

    m = PyModule_Create(&coremodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyLoggerStateType);
    if (PyModule_AddObject(m, "LoggerState", (PyObject *) &PyLoggerStateType) < 0) {
        Py_DECREF(&PyLoggerStateType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}