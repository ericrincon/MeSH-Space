from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers.core import Activation
from keras.optimizers import SGD

class NeuralNet:
    def __init__(self):
        self.model = self.create_model()
    def create_model(self, feature_size):
        model = Sequential()

        return model

    def train(self, optim_algo='sgd'):
        optimization_algo = None

        if optim_algo == 'sgd':
            optimization_algo = SGD(lr=.01, decay=1e-6, momentum=.5, nesterov=True)
        self.model.compile()
