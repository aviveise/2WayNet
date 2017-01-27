import ConfigParser

import cPickle

import lasagne
import matplotlib
import os
import sys

import numpy

from Layers.TiedDropoutLayer import TiedDropoutLayer
from testing import test_model

matplotlib.use('Agg')


from collections import OrderedDict
from theano import theano
from MISC.container import Container
from MISC.logger import OutputLog
from MISC.utils import ConfigSectionMap
from params import Params

import DataSetReaders

OUTPUT_DIR = r'/path/to/results/'
INPUT_PATH = r'/path/to/model'
VALIDATE_ALL = False
MEMORY_LIMIT = 8000000.

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    if len(sys.argv) > 2:
        top = int(sys.argv[2])
    else:
        top = 0

    model_results = {'train': [], 'validate': []}

    results_folder = os.path.join(os.getcwd(), 'results')

    OutputLog().set_path(results_folder)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    Params.print_params()

    # Export network
    path = OutputLog().output_path

    x_test = data_set.testset[0]
    y_test = data_set.testset[1]

    model_x = cPickle.load(open(os.path.join(INPUT_PATH, 'model_x.p'), 'rb'))
    model_y = cPickle.load(open(os.path.join(INPUT_PATH, 'model_y.p'), 'rb'))

    x_var = model_x[0].input_var
    y_var = model_y[0].input_var

    hidden_x = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_x)
    hidden_y = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_y)
    hidden_y = list(reversed(hidden_y))

    hooks = OrderedDict()

    test_y = theano.function([x_var],
                             [lasagne.layers.get_output(hidden_x[Params.OUTPUT_LAYER], moving_avg_hooks=hooks,
                                                        deterministic=True)],
                             on_unused_input='ignore')
    test_x = theano.function([y_var],
                             [lasagne.layers.get_output(hidden_y[Params.OUTPUT_LAYER], moving_avg_hooks=hooks,
                                                        deterministic=True)],
                             on_unused_input='ignore')

    batch_number = data_set.trainset[0].shape[0] / Params.BATCH_SIZE

    test_model(test_x, test_y, x_test, y_test, preprocessors=data_set.preprocessors, reduce=data_set.reduce_val)