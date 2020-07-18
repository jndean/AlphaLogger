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
    Position positions[4];
    int8_t scores[4];
    int8_t protesters[4];

	int8_t board[5 * 5 * 4];
	int8_t unoccupied[5 * 5];
	int8_t legal_moves[5 * 5 * 10];

    uint8_t num_players;
    uint8_t num_unprotested_trees;
    uint8_t current_player;

} LoggerState;


/* 
  Initialise the memory for a LoggerState struct.
  Doesn't set the memory to a valid game state.

int LoggerState_init(LoggerState* state, uint8_t num_players) {

    state->num_players = num_players;
    state->unoccupied = malloc(25 * (1 + 10 + 4 + 3 * num_players));
    if (state->unoccupied == NULL)
    	return 0;
    return 1;
}

  Uninitialise the memory in a LoggerState struct. 
  Doesn't free the LoggerState struct.

void LoggerState_uninit(LoggerState* state) {
	if (state->unoccupied != NULL) 
		free(state->unoccupied);
}
*/

/* 
  Set the game state to a new game with random player positions
*/
void LoggerState_reset(LoggerState* state, uint8_t num_players) {

	// Fill with empty markers
	memset(state->scores,      0, sizeof(state->scores));
	memset(state->protesters,  1, sizeof(state->protesters));
	memset(state->board,      -1, sizeof(state->board));
	memset(state->unoccupied,  1, sizeof(state->unoccupied));

	state->num_players = num_players;
	state->num_unprotested_trees = 0;
	state->current_player = 0;

	// Place players
	int corner;
	int8_t taken_corners[4] = {0, 0, 0, 0};
	for (int p = 0; p < num_players; ++p) {
		do {corner = rand() & 3;} while(taken_corners[corner]);
		taken_corners[corner] = 1;
		int8_t x = (corner & 1) << 2;
		int8_t y = (corner & 2) << 1;
		state->positions[p].x = x;
		state->positions[p].y = y;
		state->unoccupied[5 * y + x] = 0;
	}

	// Place centre sapling
	size_t pos = (2 * 5 + 2) * 4 + 0;  // Y=2, X=2, C=0
	state->board[pos] = 1;
	state->unoccupied[pos] = 0;
}


int on_board(Position pos) {
	return (pos.x >=0) && (pos.x < 5) && (pos.y >= 0) && (pos.y < 5);
}


/* 
  Compute the next game state after the given move, and put it in the output
  state object.
*/
void LoggerState_domove(LoggerState* in, LoggerState* out, int move) {
	Position pos = {.y = move / 50, .x = (move % 50) / 10};
	int action = move % 10;

	// Move player


}

// ---------------------------- PyLoggerState wrapper ---------------------------- //

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
PyLoggerState_getarray(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{
	LoggerState* state = self->state;

	const int num_channels = 4 + 3 * state->num_players;
	npy_intp dims[] = {5, 5, num_channels};
	PyObject *out_arr = PyArray_SimpleNew(3, dims, NPY_INT8);

	int8_t* out_data = PyArray_GETPTR1((PyArrayObject*) out_arr, 0);
	for(int xy = 0; xy < 25; ++xy) {
		int8_t* in = state->board + xy * 4;
		int8_t* out = out_data + xy * num_channels;
		*(out++) = *(in++);
		*(out++) = *(in++);
		*(out++) = *(in++);
		*(out++) = *(in);
		for (int p = 0; p < state->num_players; ++p) {
			int p_actual = (p + state->current_player) % state->num_players;
			*(out++) = -1;
			*(out++) = state->scores[p_actual];
			*(out++) = state->protesters[p_actual];
		}
	}

	for (int p = 0; p < state->num_players; ++p) {
		Position pos = state->positions[(p + state->current_player) % state->num_players];
		out_data[(5 * pos.y + pos.x) * num_channels + 4] = 1;
	}

	return out_arr;
}


static PyMethodDef PyLoggerState_methods[] = {
    {"get_array", (PyCFunction) PyLoggerState_getarray, METH_NOARGS,
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