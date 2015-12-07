__author__ = 'ericrincon'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.core import Activation
from keras.optimizers import SGD




def create_model(input_dimension, hidden_layer_size, n_classes, dropout=False, dropout_p=.5, activation_function='relu'):
    model = Sequential()

    # Input shape: 2D tensor (number of samples, input dimension).
    # Output shape: 2D tensor (number of samples, output dimension).
    model.add(Dense(
        input_dim=input_dimension,
        output_dim=hidden_layer_size,
        init='lecun_uniform'
    ))
    model.add(Activation(activation_function))

    if dropout:
        model.add(model.add(Dropout(dropout_p)))

    model.add(Dense(
        input_dim=hidden_layer_size,
        output_dim=hidden_layer_size,
        init='lecun_uniform'
    ))
    model.add(Activation(activation_function))

    model.add(Dense(
        input_dim=hidden_layer_size,
        output_dim=n_classes,
        init='lecun_uniform'
    ))
    # Output dimension: (number of samples, dimension size)
    model.add(Activation('softmax'))

    sgd = SGD(
        lr=0.1,
        decay=1e-6,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy'
    )



    return model

