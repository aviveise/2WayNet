import os
import numpy
from matplotlib import pyplot
from tabulate import tabulate

from MISC.logger import OutputLog
from MISC.utils import complete_rank, euclidean_error, calculate_correlation
from params import Params


def iterate_single_minibatch(inputs, batchsize, shuffle=False, preprocessor=None):
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if preprocessor is not None:
            yield preprocessor(numpy.copy(inputs[excerpt]))
        else:
            yield inputs[excerpt]


def test_model(model_x, model_y, dataset_x, dataset_y, preprocessors=None, reduce=0):
    test_x = dataset_x
    test_y = dataset_y

    x_total_value = None
    y_total_value = None

    if preprocessors is None:
        preprocessors = (None, None)

    for index, batch in enumerate(
            iterate_single_minibatch(test_x, Params.VALIDATION_BATCH_SIZE, False, preprocessor=preprocessors[0])):
        x_values = model_y(batch)[0]

        if x_total_value is None:
            x_total_value = x_values
        else:
            x_total_value = numpy.vstack((x_total_value, x_values))

    for index, batch in enumerate(
            iterate_single_minibatch(test_y, Params.VALIDATION_BATCH_SIZE, False, preprocessor=preprocessors[1])):

        y_values = model_x(batch)[0]

        if y_total_value is None:
            y_total_value = y_values
        else:
            y_total_value = numpy.vstack((y_total_value, y_values))

    for index, (x_tilde, y_tilde) in enumerate(zip(y_total_value, x_total_value)):
        x_tilde_reshape = x_tilde.reshape((28, 14), order='F')
        y_tilde_reshape = y_tilde.reshape((28, 14), order='F')

        x_reshape = test_x[index].reshape((28, 14), order='F')
        y_reshape = test_y[index].reshape((28, 14), order='F')

        image_tilde_x = numpy.hstack((x_tilde_reshape, y_reshape))
        image_tilde_y = numpy.hstack((x_reshape, y_tilde_reshape))

        image_tilde_x = (image_tilde_x + abs(image_tilde_x)) / 2
        image_tilde_y = (image_tilde_y + abs(image_tilde_y)) / 2

        pyplot.imsave(os.path.join('/home/avive/theses/MNIST_results2/x/', '{0}.jpg'.format(index)), image_tilde_x,
                      cmap='Greys_r')
        pyplot.imsave(os.path.join('/home/avive/theses/MNIST_results2/y/', '{0}.jpg'.format(index)), image_tilde_y,
                      cmap='Greys_r')

    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'desc1', 'desc5']

    rows = []

    search_recall, describe_recall = complete_rank(x_total_value, y_total_value, reduce)

    loss = euclidean_error(x_total_value, y_total_value)
    correlation = calculate_correlation(x_total_value, y_total_value)

    print_row = ["{0} ".format(Params.OUTPUT_LAYER), loss, correlation]
    print_row.extend(search_recall)
    print_row.extend(describe_recall)

    rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))
