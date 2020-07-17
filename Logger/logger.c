#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>

#include<Python.h>
#include <numpy/arrayobject.h>

#define ALLOC_ERROR() do {printf("alloc error\n"); exit(1701);} while(0);



// ---------------------------- LoggerState ---------------------------- //
/*
For P players, LoggerState contains 5 + 3P layers of 5x5 signed bytes.
Positions are marked with 1, empty spaces with -1 layers passed to the NN
The layers are:
- Occupation (not passed to NN, 1 for occupied, 0 elsewhere)
- Saplings
- Young trees
- Mature trees
- Protesters
- For each Player:
  - Location
  - Score
  - Num protesters
*/

typedef signed char Layer[25];

typedef struct {
    unsigned char num_players;
    Layer* layers;
} LoggerState;



int LoggerState_init(LoggerState* state, unsigned char num_players) {

    state->num_players = num_players;
    state->layers = malloc((5 + 3 * num_players) * sizeof(Layer));
    if (state->layers == NULL)
    	return 0;

    return 1;
}

void LoggerState_uninit(LoggerState* state) {
	if (state->layers != NULL) free(state->layers);
}

void LoggerState_reset(LoggerState* state) {
	memset(state->layers[0], 0, sizeof(Layer));
	memset(state->layers[1], -1, 4 * sizeof(Layer));
	for (int p = 0; p < state->num_players; ++p) {
		memset(state->layers[5 + 3 * p], 0, 2 * sizeof(Layer));
		memset(state->layers[5 + 3 * p + 2], 1, sizeof(Layer));
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
	npy_intp dims[] = {4 + 3 * self->content.num_players, 5, 5};
	PyObject *arr = PyArray_SimpleNewFromData(
		3, dims, NPY_INT8, self->content.layers[1]
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