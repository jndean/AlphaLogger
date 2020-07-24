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
LoggerState array format:
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


typedef struct {
  int8_t y;
  int8_t x;
} Vec2;


Vec2 DIRECTIONS[] = {
  {-1,  0},
  { 0, -1},
  { 0,  1},
  { 1,  0},
};

Vec2 MOTIONS[] = {
  {-2,  0},
  {-1, -1}, {-1,  0}, {-1,  1},
  { 0, -2}, { 0, -1}, { 0,  0}, {0, 1}, {0, 2},
  { 1, -1}, { 1,  0}, { 1,  1},
  { 2,  0},
};


typedef struct {
  Vec2 positions[4];
  int8_t scores[4];
  int8_t protesters[4];

  int8_t board[5 * 5 * 4];
  int8_t unoccupied[5 * 5];
  int8_t legal_moves[5 * 5 * 10];

  uint8_t num_players;
  uint8_t num_unprotested_trees;
  uint8_t current_player;

} LoggerState;


void _update_legal_moves(LoggerState* state);


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

  _update_legal_moves(state);
}


#define ON_BOARD(px, py) ((px >=0) && (px < 5) && (py >= 0) && (py < 5))


typedef struct {
  int8_t y, x, action, protest_y, protest_x;
} Move;


void _grow(LoggerState* state);
void _plant(LoggerState* state, int direction_idx);
void _chop(LoggerState* state, int direction_idx);
void _protest(LoggerState* state, int8_t y, int8_t x);

/* 
  Advance the game state according to the current player making the given move
*/
int LoggerState_domove(LoggerState* state, Move move) {
  
  // Move player
  Vec2* pos = &state->positions[state->current_player];
  state->unoccupied[5 * pos->y + pos->x] = 1;
  pos->y = move.y;
  pos->x = move.x;
  state->unoccupied[5 * move.y + move.x] = 0;

  _grow(state);

  // Do action
  if (move.action < 4) {
    _chop(state, move.action);
  } else if (move.action < 8) {
    _plant(state, move.action - 4);
  } else if (move.action == 8) {
    _protest(state, move.protest_y, move.protest_x);
  } 
  // Action 9 is a pass that does nothing

  // Check for winner
  for (int p = 0; p < state->num_players; ++p) {
    if (state->scores[p] >= 10) 
      return p;
  }

  // Move to next turn
  state->current_player = (state->current_player + 1) % state->num_players;
  _update_legal_moves(state);

  return -1;  // No winner
}


void __grow_square(LoggerState* state, int8_t* new_saplings, int8_t y, int8_t x) {
  // Pointer to the given square in the board
  int8_t* square = state->board + (5 * y + x) * 4;

  if (square[SAPLINGS] == 1) {
    if (new_saplings[5 * y + x])
      return;  // This sapling spawned this turn, so doesn't grow
    square[SAPLINGS] = -1;
    square[YOUNGTREES] = 1;
  }
  else if (square[YOUNGTREES] == 1) {
    square[YOUNGTREES] = -1;
    square[MATURETREES] = 1;
    state->num_unprotested_trees += 1;
  }
  else if (square[MATURETREES] == 1) {
    // Remove this repetition
    for (int i = 0; i < 4; ++i) {
      Vec2 direction = DIRECTIONS[i];
      int8_t sq_x = x + direction.x;
      int8_t sq_y = y + direction.y;
      int8_t sq_yx = 5 * sq_y + sq_x;
      if (ON_BOARD(sq_x, sq_y) && state->unoccupied[sq_yx]) {
        state->board[sq_yx * 4 + SAPLINGS] = 1;
        state->unoccupied[sq_yx] = 0;
        new_saplings[sq_yx] = 1;
      }
    }
  }
}

void _grow(LoggerState* state) {
  int8_t new_saplings[5 * 5] = {0}; // Mark saplings that spawned this turn and so don't grow

  Vec2 player_pos = state->positions[state->current_player];
  for (int8_t i = 0; i < 5; ++i) {
    __grow_square(state, new_saplings, i, player_pos.x);
    __grow_square(state, new_saplings, player_pos.y, i);
  }
}

void _plant(LoggerState* state, int direction_idx) {
  Vec2 player_pos = state->positions[state->current_player];
  Vec2 direction = DIRECTIONS[direction_idx];
  int8_t y = player_pos.y + direction.y;
  int8_t x = player_pos.x + direction.x;
  int8_t yx = 5 * y + x;
  state->board[yx * 4 + SAPLINGS] = 1;
  state->unoccupied[yx] = 0;
}

void _chop(LoggerState* state, int direction_idx) {
  int current_player = state->current_player;
  Vec2 square = state->positions[current_player];
  Vec2 direction = DIRECTIONS[direction_idx];
  while (1) {
    square.y += direction.y;
    square.x += direction.x;
    int8_t yx = 5 * square.y + square.x;
    int8_t yx4 = yx * 4;
    if (!(ON_BOARD(square.y, square.x) && state->board[yx4 + MATURETREES] == 1))
      break;
    state->board[yx4 + MATURETREES] = -1;
    state->unoccupied[yx] = 1;
    state->scores[current_player] += 1;
    if (state->board[yx4 + PROTESTERS] == 1) {
      state->board[yx4 + PROTESTERS] = -1;
      state->protesters[current_player] += 1;
    } else {
      state->num_unprotested_trees -= 1;
    }
  }
}

void _protest(LoggerState* state, int8_t y, int8_t x) {
  state->board[(5 * y + x) * 4 + PROTESTERS] = 1;
  state->protesters[state->current_player] -= 1;
  state->num_unprotested_trees -= 1;
}


void _update_legal_moves(LoggerState* state) {

  // Temporarily unoccupy current space. Reset at end of function
  Vec2 player_pos = state->positions[state->current_player];
  int player_yx = player_pos.y * 5 + player_pos.x;
  state->unoccupied[player_yx] = 1;

  memset(state->legal_moves, 0, sizeof(state->legal_moves));

  int8_t can_protest = state->num_unprotested_trees > 0 
                     && state->protesters[state->current_player] > 0;

  // For a motion to be legal, the destination must be unoccupied and on the board.
  int8_t legal_motions[13];
  for (int i = 0; i < 13; ++i) {
    Vec2 pos = MOTIONS[i];
    pos.y += player_pos.y;
    pos.x += player_pos.x;
    legal_motions[i] = ON_BOARD(pos.y, pos.x) && state->unoccupied[5 * pos.y + pos.x];
  }

  // Additionally, the two-square motions require the intermediate square to be legal
  legal_motions[1] &= legal_motions[2] || legal_motions[5];
  legal_motions[3] &= legal_motions[2] || legal_motions[7];
  legal_motions[9] &= legal_motions[5] || legal_motions[10];
  legal_motions[11] &= legal_motions[7] || legal_motions[10];
  legal_motions[0] &= legal_motions[2];
  legal_motions[4] &= legal_motions[5];
  legal_motions[8] &= legal_motions[7];
  legal_motions[12] &= legal_motions[10];

  // For each legal motion, compute the subsequent legal actions
  for (int i = 0; i < 13; ++i) {
    if (!legal_motions[i]) 
      continue;
    Vec2 pos = MOTIONS[i];
    pos.y += player_pos.y;
    pos.x += player_pos.x;
    int pos_yx = (5 * pos.y + pos.x) * 10;

    int8_t actions_available = can_protest;

    // Chops and plants can happen in 4 directions
    for (int d_i = 0; d_i < 4; ++d_i) {
      Vec2 direction = DIRECTIONS[d_i];
      Vec2 square = direction;
      square.x += pos.x;
      square.y += pos.y;
      if (!ON_BOARD(square.y, square.x))
        continue;
      Vec2 next_square = square;
      next_square.x += direction.x;
      next_square.y += direction.y;

      int sq_yx = 5 * square.y + square.x;
      int sq_yx4 = sq_yx * 4;

      // Can chop if there's an unprotested mature tree or a young tree that will grow
      if (state->board[sq_yx4 + YOUNGTREES] == 1
          || (state->board[sq_yx4 + MATURETREES] == 1 && state->board[sq_yx4 + PROTESTERS] != 1)) {
        state->legal_moves[sq_yx * 10 + d_i] = 1;
        actions_available = 1;
      } 

      // Can plant if the square is empty
      // (The square after can't be mature, or a new sapling will spawn here during _grow)
      else if (state->unoccupied[sq_yx]
               && (!ON_BOARD(next_square.y, next_square.x) 
                   || state->board[(5 * next_square.y + next_square.x) * 4 + MATURETREES] != 1)) {
        state->legal_moves[sq_yx * 10 + 4 + d_i] = 1;
        actions_available = 1;
      }
    }

    // Can play a protester if you have one and there's a suitable tree (not square dependent)
    state->legal_moves[pos_yx + 8] = can_protest;

    // Not actions available in this position, so pass is legal
    state->legal_moves[pos_yx + 9] = !actions_available;
  }

  // Reset temporary change to occupation of current space
  state->unoccupied[player_yx] = 0;
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
PyLoggerState_getstatearray(PyLoggerState *self, PyObject *Py_UNUSED(ignored)) 
{
  LoggerState* state = self->state;

  const int num_channels = 4 + 3 * state->num_players;
  npy_intp dims[] = {5, 5, num_channels};
  PyObject *out_arr = PyArray_SimpleNew(3, dims, NPY_INT8);
  if (out_arr == NULL) 
    return NULL;

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
    Vec2 pos = state->positions[(p + state->current_player) % state->num_players];
    out_data[(5 * pos.y + pos.x) * num_channels + 4 + 3 * p] = 1;
  }

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

  Move move = {.y = 3, .x = 2, .action = 4, .protest_y = 0, .protest_x = 0};
  LoggerState_domove(self->state, move);
  for (int i=0; i < 10; ++i){
    move = (Move){.y = 3, .x = 2, .action = 9, .protest_y = 0, .protest_x = 0};
    LoggerState_domove(self->state, move);
  }
  move = (Move){.y = 3, .x = 2, .action = 0, .protest_y = 0, .protest_x = 0};
  LoggerState_domove(self->state, move);
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