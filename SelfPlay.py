
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Input, Flatten, Dense, Lambda
)
from tensorflow.keras.models import Model
from tqdm import tqdm

import Game
import Player


def self_play_matches(player, num_players=2, num_samples=100, max_turns=100, silent=False):

    board = Game.Board(num_players)
    board.reset()
    player.reset()
    states = np.empty((num_samples+max_turns,) + board.get_state().shape, dtype=np.int8)
    probs = np.empty((num_samples+max_turns, board.num_moves), dtype=np.float32)
    scores = np.empty((num_samples+max_turns, num_players), dtype=np.float32)

    last_reset = 0
    _tqdm = (lambda x: x) if silent else tqdm
    for i in _tqdm(range(num_samples + max_turns)):
        move = player.choose_move(board)

        states[i] = board.get_state()
        probs[i] = player.MCTS.root_node.mcts_probs

        result = board.do_move(move)
        player.done_move(move)

        if result is not None:
            # Retroactively set scores for the whole match
            turn_offset = (i - last_reset) % num_players
            for pid in range(num_players):
                scores[last_reset + pid: i + 1: num_players] = np.roll(result, turn_offset - pid)

            if i >= num_samples:
                break
            last_reset = i
            board.reset()
            player.reset()

    return states[:num_samples], probs[:num_samples], scores[:num_samples]


def create_model(
        input_shape, num_moves, num_players,
        num_resnet_blocks=20,
        conv_filters=256,
        value_dense_neurons=256,
        lr_momentum=0.9,
        l2_regularisation=0.0001
):

    L2 = tf.keras.regularizers.l2(l=l2_regularisation)
    dense_params = {"kernel_regularizer": L2, "bias_regularizer": L2}
    conv_params = {"kernel_regularizer": L2, "bias_regularizer": L2,
                   "data_format": "channels_first"}

    input_ = Input(shape=input_shape)

    # The feature extractor body
    feature_output = Conv2D(conv_filters, 3, padding='same', **conv_params)(input_)
    feature_output = BatchNormalization()(feature_output)
    feature_output = ReLU()(feature_output)
    skip_connection = feature_output
    for _ in range(num_resnet_blocks):
        feature_output = Conv2D(conv_filters, 3, padding='same', **conv_params)(feature_output)
        feature_output = BatchNormalization()(feature_output)
        feature_output = ReLU()(feature_output)
        feature_output = Conv2D(conv_filters, 3, padding='same', **conv_params)(feature_output)
        feature_output = BatchNormalization()(feature_output)
        feature_output = feature_output + skip_connection
        feature_output = ReLU()(feature_output)
        skip_connection = feature_output

    # The Value head
    value_output = Conv2D(1, 1, padding='same', **conv_params)(feature_output)
    value_output = BatchNormalization()(value_output)
    value_output = ReLU()(value_output)
    value_output = Flatten()(value_output)
    value_output = Dense(value_dense_neurons, **dense_params)(value_output)
    value_output = ReLU()(value_output)
    value_output = Dense(num_players, activation='softmax', **dense_params)(value_output)
    value_output = 2. * value_output - 1  # Transform from [0, 1] to [-1, 1]
    value_output = Lambda(lambda x: x, name="value_output")(value_output)

    # The Policy head
    policy_output = Conv2D(2, 1, padding='same', **conv_params)(feature_output)
    policy_output = BatchNormalization()(policy_output)
    policy_output = ReLU()(policy_output)
    policy_output = Flatten()(policy_output)
    policy_output = Dense(
        num_moves, activation='softmax', name='policy_output', **dense_params)(policy_output)

    model = Model(inputs=input_, outputs=[value_output, policy_output])
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=lr_momentum)
    losses = {'value_output': 'mse', 'policy_output': 'bce'}
    loss_weights = {'value_output': 1.0, 'policy_output': 1.0}
    model.compile(optimizer=optimiser, loss=losses, loss_weights=loss_weights)

    return model


if __name__ == "__main__":

    num_players = 2
    game = Game.Board(num_players)
    game.reset()

    print('Building model')
    model = create_model(
        input_shape=game.get_state().shape,
        num_moves=game.num_moves,
        num_players=num_players,
        num_resnet_blocks=20,
        conv_filters=256,
        value_dense_neurons=256,
        lr_momentum=0.9,
        l2_regularisation=0.0001
    )

    print("Self-play")
    player = Player.RandomMCTS(id_="Player", simulations_per_turn=50, max_rollout=30, learning=True)
    states, probs, scores = self_play_matches(player, 2, num_samples=40)

    print('Training')
    model.fit(
        states, [scores, probs]
    )