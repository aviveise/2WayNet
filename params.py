import lasagne
from MISC.logger import OutputLog
from Layers.LocallyDenseLayer import TiedDenseLayer
from Layers.TiedDropoutLayer import TiedDropoutLayer


class Params:
    """
    Parameters for the training and inference of the 2-WayNet
    """
    # region Training Params
    BATCH_SIZE = 128  # number of samples in the batch for training
    VALIDATION_BATCH_SIZE = 1000  # number of samples in the batch for testing
    CROSS_VALIDATION = True  # enable the running on validation after each epoch
    EPOCH_NUMBER = 100  # number of epochs
    DECAY_EPOCH = [20, 40, 60, 80]  # epochs which include a learning rate decay
    DECAY_RATE = 0.5  # The factor to multiply the learning rate in each decay
    BASE_LEARNING_RATE = 0.0001  # starting learning rate
    MOMENTUM = 0.9  # momentum for the training
    # endregion

    # region Loss Weights
    # Coefficients for the loss and regularization terms
    WEIGHT_DECAY = 0.05
    GAMMA_COEF = 0.05
    WITHEN_REG_X = 0.05
    WITHEN_REG_Y = 0.05
    L2_LOSS = 1
    LOSS_X = 1
    LOSS_Y = 1
    # endregion

    # region Architecture
    LAYER_SIZES = [392, 50, 392]  # Size of the hidden layers
    OUTPUT_LAYER = 1  # The layer from which to take the representations
    DROP_PROBABILITY = 0.5  # Probability for removing a neuron in the dropout/tied dropout layer
    WEIGHT_INIT = lasagne.init.GlorotUniform()  # Initialization method for the weights
    LAYER_TYPES = [TiedDenseLayer, TiedDenseLayer, TiedDenseLayer,
                   TiedDenseLayer]  # Types of layers can be TiedDenseLayer or LocallyDenseLayer
    LEAKINESS = 0.3  # Leakiness coefficient
    LOCALLY_DENSE_M = 2  # The number of sub-dense layer in the locally dense layer
    NOISE_LAYER = TiedDropoutLayer  # The type of dropout layer can be TiedDropoutLayer or Dropoutlayer
    BN = True  # If True uses batch normalization
    BN_ACTIVATION = False  # Controls the order of non-linearity, if True the non-linearity is performed after the BN
    SIMILARITY_METRIC = 'correlation'  # controls the type of distance metric to use in calculating matching

    # endregion

    @classmethod
    def print_params(cls):
        OutputLog().write('Params:\n')
        for (key, value) in cls.__dict__.iteritems():
            if not key.startswith('__'):
                OutputLog().write('{0}: {1}'.format(key, value))
