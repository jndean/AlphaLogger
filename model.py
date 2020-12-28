import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Input, Flatten, Dense, Lambda
)
from tensorflow.keras.models import Model



def create_model(
        input_shape, num_moves, num_players,
        num_resnet_blocks=20,
        conv_filters=256,
        value_dense_neurons=256,
        lr_momentum=0.9,
        l2_regularisation=0.0001
):

    L2 = tf.keras.regularizers.l2(l=l2_regularisation)
    regularisers = {"kernel_regularizer": L2, "bias_regularizer": L2}

    input_ = Input(shape=input_shape)

    # The feature extractor body
    feature_output = Conv2D(conv_filters, 3, padding='same', **regularisers)(input_)
    feature_output = BatchNormalization()(feature_output)
    feature_output = ReLU()(feature_output)
    skip_connection = feature_output
    for _ in range(num_resnet_blocks):
        feature_output = Conv2D(conv_filters, 3, padding='same', **regularisers)(feature_output)
        feature_output = BatchNormalization()(feature_output)
        feature_output = ReLU()(feature_output)
        feature_output = Conv2D(conv_filters, 3, padding='same', **regularisers)(feature_output)
        feature_output = BatchNormalization()(feature_output)
        feature_output = feature_output + skip_connection
        feature_output = ReLU()(feature_output)
        skip_connection = feature_output

    # The Value head
    value_output = Conv2D(1, 1, padding='same', **regularisers)(feature_output)
    value_output = BatchNormalization()(value_output)
    value_output = ReLU()(value_output)
    value_output = Flatten()(value_output)
    value_output = Dense(value_dense_neurons, **regularisers)(value_output)
    value_output = ReLU()(value_output)
    value_output = Dense(num_players, activation='softmax', **regularisers)(value_output)
    value_output = 2. * value_output - 1  # Transform from [0, 1] to [-1, 1]
    value_output = Lambda(lambda x: x, name="value_output")(value_output)

    # The Policy head
    policy_output = Conv2D(2, 1, padding='same', **regularisers)(feature_output)
    policy_output = BatchNormalization()(policy_output)
    policy_output = ReLU()(policy_output)
    policy_output = Flatten()(policy_output)
    policy_output = Dense(
        num_moves, activation='softmax', name='policy_output', **regularisers)(policy_output)

    model = Model(inputs=input_, outputs=[policy_output, value_output])
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=lr_momentum)
    losses = {'value_output': 'mse', 'policy_output': 'bce'}
    loss_weights = {'value_output': 1.0, 'policy_output': 1.0}
    model.compile(optimizer=optimiser, loss=losses, loss_weights=loss_weights)

    return model