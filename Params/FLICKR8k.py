import lasagne

from MISC.logger import OutputLog
from Layers.LocallyDenseLayer import TiedDenseLayer
from Layers.TiedDropoutLayer import TiedDropoutLayer


class Params:

    # region Training Params
    BATCH_SIZE = 128
    VALIDATION_BATCH_SIZE = 1000
    CROSS_VALIDATION = True
    EPOCH_NUMBER = 100
    DECAY_EPOCH = [20, 40, 60, 80]
    DECAY_RATE = 0.5
    BASE_LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    # endregion

    # region Loss Weights
    WEIGHT_DECAY = 0.005
    GAMMA_COEF = 0.05
    WITHEN_REG_X = 0.5
    WITHEN_REG_Y = 0.5
    L2_LOSS = 0.25
    LOSS_X = 1
    LOSS_Y = 1
    # endregion

    # region Architecture
    LAYER_SIZES = [2000, 3000, 4000]
    TEST_LAYER = 1
    DROP_PROBABILITY = [0.5, 0.5, 0.5]
    WEIGHT_INIT = lasagne.init.GlorotUniform()
    LAYER_TYPES = [TiedDenseLayer, TiedDenseLayer, TiedDenseLayer, TiedDenseLayer]
    LEAKINESS = 0.3
    LOCALLY_DENSE_M = 2
    NOISE_LAYER = TiedDropoutLayer
    BN = True
    BN_ACTIVATION = True
    SIMILARITY_METRIC='correlation'
    # endregion

    @classmethod
    def print_params(cls):
        OutputLog().write('Params:\n')
        for (key, value) in cls.__dict__.iteritems():
            if not key.startswith('__'):
                OutputLog().write('{0}: {1}'.format(key, value))
