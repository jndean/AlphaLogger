#ifndef SELFPLAY_H
#define SELFPLAY_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ALPHALOGGER_PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY

#include <omp.h>

#include<Python.h>
#include <numpy/arrayobject.h>

#include "MCTS.h"


PyObject* self_play(PyObject* inference_method, int num_samples, int num_simulations);



#endif  /* SELFPLAY_H */