from math import ceil
from scipy.spatial.distance import cdist

import matplotlib
import traceback

from testing import test_model

matplotlib.use('Agg')

import ConfigParser
import os
import sys
import cPickle
import lasagne
import numpy

from collections import OrderedDict
from theano import tensor, theano
from MISC.container import Container
from MISC.logger import OutputLog
from MISC.utils import ConfigSectionMap, batch_normalize_updates
from Models import tied_dropout_iterative_model
from params import Params

import DataSetReaders

OUTPUT_DIR = r'/path/to/output'
VALIDATE_ALL = False
MEMORY_LIMIT = 8000000.


def iterate_parallel_minibatches(inputs_x, inputs_y, batchsize, shuffle=False, preprocessors=None):
    assert len(inputs_x) == len(inputs_y)
    if shuffle:
        indices = numpy.arange(len(inputs_x))
        numpy.random.shuffle(indices)

    batch_limit = ceil(MEMORY_LIMIT / (inputs_x.shape[1] + inputs_y.shape[1]) / batchsize / 8.)

    if not isinstance(inputs_x, numpy.ndarray):
        buffer_x = numpy.load(inputs_x.filename, 'r')

    if not isinstance(inputs_y, numpy.ndarray):
        buffer_y = numpy.load(inputs_y.filename, 'r')

    for start_idx in range(0, len(inputs_x) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if (start_idx / batchsize) % batch_limit == 0:
            if not isinstance(inputs_x, numpy.ndarray):
                buffer_x = numpy.load(inputs_x.filename, 'r')

            if not isinstance(inputs_y, numpy.ndarray):
                buffer_y = numpy.load(inputs_y.filename, 'r')

        if preprocessors is not None:
            yield preprocessors[0](numpy.copy(buffer_x[excerpt])), \
                  preprocessors[1](numpy.copy(buffer_y[excerpt]))
        else:
            yield buffer_x[excerpt], buffer_y[excerpt]


def iterate_single_minibatch(inputs, batchsize, shuffle=False, preprocessor=None):
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)

    batch_limit = ceil(MEMORY_LIMIT / inputs.shape[1] / batchsize / 4.)

    buffer = numpy.load(inputs.filename, 'r')

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if (start_idx / batchsize) % batch_limit == 0:
            buffer = numpy.load(inputs.filename, 'r')

        if preprocessor is not None:
            yield preprocessor(numpy.copy(buffer[excerpt]))
        else:
            yield buffer[excerpt]


if __name__ == '__main__':

    data_set_config = sys.argv[1]

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

    y_var = tensor.matrix()
    x_var = tensor.matrix()

    model = tied_dropout_iterative_model

    Params.print_params()

    OutputLog().write('Model: {0}'.format(model.__name__))

    # Export network
    path = OutputLog().output_path

    x_train = data_set.trainset[0]
    y_train = data_set.trainset[1]

    model_x, model_y, hidden_x, hidden_y, loss, outputs, hooks = model.build_model(x_var,
                                                                                   x_train.shape[1],
                                                                                   y_var,
                                                                                   y_train.shape[1],
                                                                                   layer_sizes=Params.LAYER_SIZES,
                                                                                   weight_init=Params.WEIGHT_INIT)

    params_x = lasagne.layers.get_all_params(model_x, trainable=True)
    params_y = lasagne.layers.get_all_params(model_y, trainable=True)

    if hooks:
        updates = OrderedDict(batch_normalize_updates(hooks, 100))
    else:
        updates = OrderedDict()

    params_x.extend(params_y)

    params = lasagne.utils.unique(params_x)

    current_learning_rate = Params.BASE_LEARNING_RATE

    updates.update(
        lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_learning_rate, momentum=Params.MOMENTUM))

    train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    inference_model_y = theano.function([x_var],
                                        [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True)
                                         for layer in
                                         hidden_x],
                                        on_unused_input='ignore')
    inference_model_x = theano.function([y_var],
                                        [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True)
                                         for layer in
                                         hidden_y],
                                        on_unused_input='ignore')

    batch_number = data_set.trainset[0].shape[0] / Params.BATCH_SIZE

    output_string = '{0}/{1} loss: {2} '
    output_string += ' '.join(['{0}:{{{1}}}'.format(key, index + 3) for index, key in enumerate(outputs.keys())])

    for epoch in range(Params.EPOCH_NUMBER):
        OutputLog().write('Epoch {0}'.format(epoch))

        model_results['train'].append({'loss': []})
        model_results['validate'].append({})

        for label in outputs.keys():
            model_results['train'][epoch][label] = []

        for index, batch in enumerate(
                iterate_parallel_minibatches(x_train, y_train, Params.BATCH_SIZE, False, data_set.preprocessors)):
            input_x, input_y = batch
            train_loss = train_fn(numpy.cast[theano.config.floatX](input_x),
                                  numpy.cast[theano.config.floatX](input_y))

            model_results['train'][epoch]['loss'].append(train_loss[0])
            for label, value in zip(outputs.keys(), train_loss[1:]):
                model_results['train'][epoch][label].append(value)

            OutputLog().write(output_string.format(index, batch_number, *train_loss))

            del batch, input_x, input_y
            del train_loss

        if Params.CROSS_VALIDATION or epoch in Params.DECAY_EPOCH:
            tuning_x = data_set.tuning[0]
            tuning_y = data_set.tuning[1]

            OutputLog().write('\nValidating model\n')

            test_model(inference_model_x, inference_model_y, tuning_x, tuning_y, preprocessors=data_set.preprocessors,
                       reduce=data_set.reduce_val)

        if epoch in Params.DECAY_EPOCH:
            current_learning_rate *= Params.DECAY_RATE
            if hooks:
                updates = OrderedDict(batch_normalize_updates(hooks, 100))
            else:
                updates = OrderedDict()

            with file(os.path.join(path, 'model_x_{0}.p'.format(epoch)), 'w') as model_x_file:
                cPickle.dump(model_x, model_x_file)

            with file(os.path.join(path, 'model_y{0}.p'.format(epoch)), 'w') as model_y_file:
                cPickle.dump(model_y, model_y_file)

            updates.update(
                lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_learning_rate,
                                                  momentum=Params.MOMENTUM))
            del train_fn
            train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    OutputLog().write('Test results')

    try:
        test_model(inference_model_x, inference_model_y, data_set.testset[0],
                   data_set.testset[1], preprocessors=data_set.preprocessors,
                   reduce=data_set.reduce_val)
    except Exception as e:
        OutputLog().write('Error testing model with exception {0}'.format(e))
        traceback.print_exc()

    with file(os.path.join(path, 'model_x.p'), 'w') as model_x_file:
        cPickle.dump(model_x, model_x_file)

    with file(os.path.join(path, 'model_y.p'), 'w') as model_y_file:
        cPickle.dump(model_y, model_y_file)

    with file(os.path.join(path, 'results.p'), 'w') as results_file:
        cPickle.dump(model_results, results_file)
