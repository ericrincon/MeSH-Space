__author__ = 'ericrincon'

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense
from keras.layers.core import RepeatVector
from keras.layers.core import Dropout

"""
    Encoder decoder implementation from paper "Sequence to Sequence Learning"
    http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

    rnn_unit_type: The type of unit to use for the RNN. LSTM or GRU
    hidden_layer_size:
    n_decoder_layers: Set to 4 as default as per the paper on page 3.

    Other relevant papers:
    Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    http://arxiv.org/pdf/1406.1078v3.pdf

    http://arxiv.org/pdf/1406.1078v3.pdf
"""

class Seq2Seq():
    def __init__(self, input_layer_size, hidden_layer_size, rnn_unit_type, n_encoder_layers=4, n_decoder_layers=4,
                 decoder_rnn_unit_type=None, activation_function='tanh', inner_activation_function='hard_sigmoid',
                 optimization_algorithm='sgd', loss_function='categorical_crossentropy',
                 dropout=True, dropout_p=.5):
        self.model = self.create_model()

    def create_model(self, input_layer_size, hidden_layer_size, rnn_unit_type, n_encoder_layers=4, n_decoder_layers=4,
                     decoder_rnn_unit_type=None, activation_function='tanh', inner_activation_function='hard_sigmoid',
                     optimization_algorithm='sgd', loss_function='categorical_crossentropy', dropout=True, dropout_p=.5):
        # Set up model as linear stack of layers.
        model = Sequential()

        # Embedding layer to add masking for variable length sequence input.
        # Input should have zeros that are will be masked out.
        model.add(Embedding(
            input_dim=input_layer_size,
            output_dim=hidden_layer_size,
            init='',
            mask_zero=True
        ))

        # Set up the encoder model

        # Input dimension: (number of samples, time steps, output dimension)
        # Output dimension: (number of samples, output dimension)
        if rnn_unit_type == 'gru':
            rnn_unit = GRU
        else:
            # Default to LSTM cell if GRU is not indicated
            rnn_unit = LSTM

        for i in range(n_encoder_layers):
            model.add(rnn_unit(
                output_dim=hidden_layer_size,
                activation=activation_function,
                inner_activation=inner_activation_function,
                return_sequences=True
            ))

            if dropout:
                model.add(Dropout(dropout_p))

        # Copy the last vector output of the RNN and use it as input for the "decoder" RNN.
        # Input dimension: (number of samples, output dimension of last layer)
        # Output dimension: (number of samples, n repeat times, output dimension)
        model.add(RepeatVector(hidden_layer_size))

        # Set up the decoder model

        # Set the type of RNN unit that the decoder will have.
        # Should be the same type in most cases but option to change it.
        if decoder_rnn_unit_type is None:
            decoder_rnn_unit = rnn_unit
        elif decoder_rnn_unit_type == 'gru':
            decoder_rnn_unit = GRU
        else:
            decoder_rnn_unit = LSTM

        for i in range(n_decoder_layers):
            model.add(decoder_rnn_unit(
                output_dim=hidden_layer_size,
                activation=activation_function,
                inner_activation=inner_activation_function,
                return_sequences=True
            ))

            if dropout:
                model.add(Dropout(dropout_p))

        # Input dimension: (number of samples, time steps, input dimension)
        model.add(TimeDistributedDense(
            output_dim=10,
            activation='softmax'
        ))
        model.compile(loss=loss_function, optimizer=optimization_algorithm)

        return model

