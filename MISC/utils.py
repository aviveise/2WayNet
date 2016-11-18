import itertools
import os
import traceback
import theano
import numpy
import numpy.linalg
import scipy.linalg
import scipy.sparse.linalg

from sklearn import preprocessing
from matplotlib import pyplot
from matplotlib.pyplot import colorbar, pcolormesh, matplotlib
from theano import tensor as T
from logger import OutputLog
from params import Params

global file_ndx
file_ndx = {}

matplotlib.pyplot.ioff()


def unitnorm_rows(M):
    if M is None:
        return

    for i in xrange(M.shape[0]):

        norm = numpy.linalg.norm(M[i, :])
        if norm != 0:
            M[i, :] /= norm

    return M


def center(M):
    """
    Centers a given matrix by subtracting each column by it's mean
    :param M: the input matrix
    :return: the centered matrix and the calculated mean for each column
    """
    if M is None:
        return

    mean = M.mean(axis=1).reshape([M.shape[0], 1])
    M -= mean * numpy.ones([1, M.shape[1]])
    return M, mean


def ConfigSectionMap(section, config):
    """
    Creates a mapping of a section of a given config
    :param section: the section
    :type section: string
    :param config: ConfigParser object
    :type config: ConfigParser
    :return:
    """
    dict1 = {}

    try:
        options = config.options(section)

    except:
        return None

    for option in options:
        try:
            dict1[option] = config.get(section, option)

        except:
            dict1[option] = None

    return dict1


def calculate_correlation(x, y, visualize=False):
    """
    Returns the total correlation between x and y, if visualize equals true then a correlation matrix is saved as image
    :param x: view x - matrix of size MxD
    :param y: view y - matrix of size MxD
    :param visualize: If true outputs an image of the correlation
    :return: the sum of correlation between x and y vectors
    """
    try:
        set_size = x.shape[0]
        dim = x.shape[1]

        x, mean_x = center(x.T)
        y, mean_y = center(y.T)

        s11 = numpy.diag(numpy.diag(numpy.dot(x, x.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
        s22 = numpy.diag(numpy.diag(numpy.dot(y, y.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
        s12 = numpy.dot(x, y.T) / (set_size - 1)

        s11_chol = scipy.linalg.sqrtm(s11)
        s22_chol = scipy.linalg.sqrtm(s22)

        s11_chol_inv = scipy.linalg.inv(s11_chol)
        s22_chol_inv = scipy.linalg.inv(s22_chol)

        mat_T = numpy.dot(numpy.dot(s11_chol_inv, s12), s22_chol_inv)

        if visualize:
            visualize_correlation_matrix(mat_T, 'correlation_mat')
            visualize_correlation_matrix(numpy.sort(mat_T, axis=1), 'correlation_mat_sorted')

        return numpy.trace(mat_T)

    except Exception as e:
        OutputLog().write('Error while calculating meridia error')
        OutputLog().write('Exception {0}'.format(e))
        traceback.print_exc()
        return 0


def euclidean_error(x, y):
    """
    Computes the mean squared euclidean distance between vectors in x and y
    :param x: matrix of size MxD
    :param y: matrix of size MxD
    :return: average distance
    """
    return numpy.mean(((x - y) ** 2).sum(axis=1))


def complete_rank(x, y, reduce_x=0, x_y_mapping=None):
    """
    Computes the matching recall between samples in x and y, the returned values are the recall@1 and recall@5, the results
    are the ratio between matching samples and total samples where a match is considered if the right sample in x is ranked
    1 for a given y, this is for the recall@1. for recall@5 a match is considered if x is ranked in the top 5.
    The same is applied for finding y given a sample from x
    :param x: samples matrix of size MxD
    :param y: samples matrix of size MxD
    :param reduce_x: the number of duplicates in X (relevent for flickr5k,30k and coc
    :param x_y_mapping: a mapping betweem the matching x samples an y samples
    :return: recall for matching sample from x to y and vice versa
    """
    try:
        if reduce_x and not x.shape[0] % reduce_x == 0:
            for i in range(0, reduce_x - x.shape[0] % reduce_x):
                x = numpy.vstack((x, x[-1, :]))
                y = numpy.vstack((y, y[-1, :]))

        num_X_samples = x.shape[0]
        num_Y_samples = y.shape[0]

        if x_y_mapping is None:
            if reduce_x:
                x = x[0:x.shape[0]:reduce_x, :]
                x_y_mapping = numpy.repeat(numpy.arange(x.shape[0]), reduce_x)
                num_X_samples = num_X_samples / reduce_x
            else:
                x_y_mapping = numpy.arange(x.shape[0])

        y_x_sim_matrix = scipy.cdist(x, y, Params.SIMILARITY_METRIC)

        recall_n_vals = [1, 5]
        num_of_recall_n_vals = len(recall_n_vals)

        x_search_recall = numpy.zeros((num_of_recall_n_vals, 1))
        describe_x_recall = numpy.zeros((num_of_recall_n_vals, 1))

        x_search_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=0)
        x_search_ranks = numpy.array(
            [numpy.where(col == x_y_mapping[index])[0] for index, col in enumerate(x_search_sorted_neighbs.T)])

        for idx, recall in enumerate(recall_n_vals):
            x_search_recall[idx] = numpy.sum(x_search_ranks <= recall)

        x_search_recall = 100 * x_search_recall / num_Y_samples

        describe_y_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=1)
        describe_y_ranks = numpy.array([numpy.where(numpy.in1d(row, numpy.where(x_y_mapping == index)[0]))[0]
                                        for index, row in enumerate(describe_y_sorted_neighbs)]).min(axis=1)

        for idx, recall in enumerate(recall_n_vals):
            describe_x_recall[idx] = numpy.sum(describe_y_ranks <= recall)

        describe_x_recall = 100 * describe_x_recall / num_X_samples

        return x_search_recall, describe_x_recall
    except Exception as e:
        OutputLog().write('Error calculating rank score with exception: {0}, {1}'.format(e, traceback.format_exc()))
        return [0, 0, 0], [0, 0, 0]


def visualize_correlation_matrix(mat, name):
    """
    Saves matrix as a picture
    :param mat: import matrix
    :param name: file nmae
    :return:
    """
    path = OutputLog().output_path
    output_file = os.path.join(path, name + '.jpg')

    if name not in file_ndx:
        file_ndx[name] = 0

    if os.path.exists(output_file):
        output_file = os.path.join(path, name + '_' + str(file_ndx[name]) + '.jpg')
        file_ndx[name] += 1

    f = pyplot.figure()
    pcolormesh(mat)
    colorbar()
    f.savefig(output_file, format='jpeg')
    f.clf()
    pyplot.close()


def batch_normalize_updates(hooks, avglen):
    """
    Produces the necessary updates for batch normalization
    :param hooks: dictionary with the new values for each parameters
    :param avglen: length of the moving average
    :return:
    """
    params = list(itertools.chain(*[i[1] for i in hooks['BatchNormalizationLayer:movingavg']]))
    tensors = list(itertools.chain(*[i[0] for i in hooks['BatchNormalizationLayer:movingavg']]))

    updates = []
    mulfac = 1.0 / avglen
    for tensor, param in zip(tensors, params):
        updates.append((param, T.cast((1.0 - mulfac) * param + mulfac * tensor, dtype=theano.config.floatX)))
    return updates
