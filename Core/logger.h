#ifndef LOGGER_H
#define LOGGER_H


#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>


#define NUM_PLAYERS 2

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


typedef struct {
  int8_t y, x, action, protest_y, protest_x;
} Move;


typedef struct LoggerState_{
  Vec2 positions[NUM_PLAYERS];
  int8_t scores[NUM_PLAYERS];
  int8_t protesters[NUM_PLAYERS];

  int8_t board[5 * 5 * 4];
  int8_t unoccupied[5 * 5];
  int8_t legal_moves[5 * 5 * 10];

  uint8_t num_unprotested_trees;
  uint8_t current_player;
  int8_t game_winner;

} LoggerState;


#define ON_BOARD(px, py) ((px >=0) && (px < 5) && (py >= 0) && (py < 5))


void LoggerState_reset(LoggerState* state);
int LoggerState_domove(LoggerState* state, Move move);
void LoggerState_getstatearray(LoggerState* state, int8_t* out_array);
void LoggerState_setpositions(LoggerState* state, Vec2* new_positions);

#endif  /* LOGGER_H */