#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>
#include<stdint.h>

#include<Python.h>
#include <numpy/arrayobject.h>

// ---------------------------- LoggerState ---------------------------- //


/*
LoggerState just allocates one big block of memory of size
and stores different stuff at different offsets such that the NN can just
read portions of it as a single contiguous input tensor
Unoccupied grid (5x5 bools) comes first, then the legal_moves tensor (5x5x10 bools)
followed by the board grid (5x5x(5+3*num_players))
The channels are:
0: saplings               (1 / -1)
1: young trees            (1 / -1)
2: mature trees           (1 / -1)
3: protesters             (1 / -1)
4 onwards: 3 channels per player
        0: position        (1 for player, -1 elsewhere)
        1: score           (whole channel holds value)
        2: num_protesters  (whole channel holds value)
*/

#define SAPLINGS    0
#define YOUNGTREES  1
#define MATURETREES 2
#define PROTESTERS  3
#define PLAYERS     4


typedef struct {
	int8_t y;
    int8_t x;
} Position;


typedef struct {
	int8_t* unoccupied;
	int8_t* legal_moves;
	int8_t* board;
	int stride;
    Position player_positions[4];
    uint8_t num_players;
    uint8_t num_unprotested_trees;

} LoggerState;


/* 
  Initialise the memory for a LoggerState struct.
  Doesn't set the memory to a valid game state.
*/
int LoggerState_init(LoggerState* state, uint8_t num_players) {

    state->num_players = num_players;
    state->stride = 4 + 3 * num_players;
    state->unoccupied = malloc(25 * (1 + 10 + 4 + 3 * num_players));
    if (state->unoccupied == NULL)
    	return 0;
    state->legal_moves = state->unoccupied + 25;
    state->board = state->unoccupied + 25 * 11;

    return 1;
}


/* 
  Uninitialise the memory in a LoggerState struct. 
  Doesn't free the LoggerState struct.
*/
void LoggerState_uninit(LoggerState* state) {
	if (state->unoccupied != NULL) 
		free(state->unoccupied);
}


/* 
  Set the game state to a new game with random player positions
*/
void LoggerState_reset(LoggerState* state) {

	memset(state->unoccupied, 1, 25);
	memset(state->board, -1, 25 * (4 + 3 * state->num_players));
	for (int xy = 0; xy < 25; ++xy) {
		int8_t *pos = state->board + state->stride * xy + PLAYERS + 1;
		for (int p = 0; p < state->num_players; ++p, pos += 3) {
			*pos = 0;
			*(pos + 1) = 1;
		}
	}

	int corner;
	char corners[4] = {0, 0, 0, 0};
	for (int p = 0; p < state->num_players; ++p) {
		do {corner = rand() & 3;} while(corners[corner]);
		corners[corner] = 1;
		int8_t x = (corner & 1) << 2;
		int8_t y = (corner & 2) << 1;
		state->player_positions[p].x = x;
		state->player_positions[p].y = y;
		state->board[(5 * y + x) * state->stride + PLAYERS + 3 * p] = 1;
	}
}


// ---------------------------- PyLoggerState wrapper ---------------------------- //

typedef struct {
    PyObject_HEAD
    LoggerState content;
} PyLoggerState;

static PyObject *
PyLoggerState_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  	PyObject *num_players_obj = NULL;
  	if(!PyArg_ParseTuple(args, "O", &num_players_obj))
    	return NULL;
    long num_players = PyLong_AsLong(num_players_obj);


    PyLoggerState *self;
    self = (PyLoggerState *) type->tp_alloc(type, 0);
    if (self == NULL || LoggerState_init(&self->content, num_players) == 0)
        return NULL;

    return (PyObject *) self;
}

static void
PyLoggerState_dealloc(PyLoggerState *self)
{
    LoggerState_uninit(&self->content);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
PyLoggerState_init(PyLoggerState *self, PyObject *args, PyObject *kwds)
{
	LoggerState_reset(&self->content);
    return 0;
}

static PyObject*
PyLoggerState_toarray(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{
	npy_intp dims[] = {5, 5, self->content.stride};
	PyObject *arr = PyArray_SimpleNewFromData(
		3, dims, NPY_INT8, self->content.board
	);

	return arr;
}


static PyMethodDef PyLoggerState_methods[] = {
    {"toarray", (PyCFunction) PyLoggerState_toarray, METH_NOARGS,
     "Get the board state as a numpy array"},
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


PyObject* logger_testmethod(PyObject* self, PyObject* args){

  PyLongObject *target = NULL, *source = NULL;
  if(!PyArg_ParseTuple(args, "OO", &target, &source))
    return NULL;

  if(!PyLong_Check(target) || !PyLong_Check(source)){
    PyErr_SetString(PyExc_ValueError, "set_int requires two integers.");
    return NULL;
  }

  printf("Done");
  
  Py_RETURN_NONE;
}

// ------------------------------ Module Setup ------------------------------ //

static PyMethodDef LoggerMethods[] = {
    {"test_method",  logger_testmethod, METH_VARARGS,
     "A test method."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef loggermodule = {
   PyModuleDef_HEAD_INIT,
   "logger",  
   NULL, 
   -1,
   LoggerMethods
};


PyMODINIT_FUNC PyInit_logger(void)
{
   	time_t t;
  	srand((unsigned) time(&t));

	import_array();

    PyObject *m;
    if (PyType_Ready(&PyLoggerStateType) < 0)
        return NULL;

    m = PyModule_Create(&loggermodule);
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