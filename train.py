from os.path import join
import sys
from time import time

import numpy as np

import core
import evaluate
import player
from model import create_model, load_model


def train_model(
    model,
    epochs,
    training_samples_per_epoch,
    training_sims_per_move=50,
    eval_sims_per_move=50,
    eval_matches=100,
    checkpoint_dir=None
):

    """
    eval_players = [
        player.RandomPlayer(
            name="Random"
        ),
        player.AlphaLoggerPlayer(
            model=model,
            name="AlphaLogger",
            num_simulations=eval_sims_per_move
        ),
    ]
    """

    for epoch in range(epochs):

        X, Y_p, Y_v = core.self_play(
            lambda state: tuple(model.predict(state)),
            training_samples_per_epoch,
            training_sims_per_move
        )
        model.fit(
            x=X,
            y={'policy_output': Y_p, 'value_output': Y_v},
            epochs=2
        )

        """
        scores = evaluate.play_matches(
            eval_players,
            num_matches=eval_matches,
            max_turns=40,
        )
        print('Eval scores:', scores)
        """

        if checkpoint_dir is not None:
            model.save(join(checkpoint_dir, f'epoch{epoch}.h5'))


if __name__ == '__main__':

    num_moves = 5 * 5 * 10
    num_state_array_channels = core.num_players * 3 + 4

    if len(sys.argv) > 1:
        print(f'Loading model: {sys.argv[1]}')
        model = load_model(sys.argv[1])
    else:
        print('Building model...')
        model = create_model(
            input_shape=(5, 5, num_state_array_channels),
            num_moves=num_moves,
            num_players=core.num_players,
            num_resnet_blocks=25,
            conv_filters=96,
            value_dense_neurons=100,
        )

    print('Starting training...')
    train_model(
        model,
        epochs=40,
        training_samples_per_epoch=50000,
        training_sims_per_move=50,
        eval_sims_per_move=50,
        eval_matches=50,
        checkpoint_dir='checkpoints',
    )
