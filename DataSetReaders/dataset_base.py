import abc
import os

import random

import cPickle

import functools
import hickle
import theano
import numpy

from sklearn import preprocessing
from sklearn.decomposition import PCA
from MISC.logger import OutputLog


class IdentityPreprocessor():
    def transform(self, x):
        return x


class TransposedPreprocessor():
    def __init__(self, preprocessor):
        self._preprocessor = preprocessor

    def transform(self, x):
        return self._preprocessor.transform(x.T).T


class DatasetBase(object):
    """
    Base dataset building class, to extend this class, one must implement the build_dataset method in child class
    In order for the container to recognize the dataset reader classes, there is a need for a __init__.py file
    containing imports for all dataset reader classes.
    """

    def __init__(self, data_set_parameters):

        OutputLog().write('Loading dataset: ' + data_set_parameters['name'])

        self.dataset_path = data_set_parameters['path']

        self.trainset = None
        self.testset = None
        self.tuning = None

        self.reduce_val = 0
        self.x_y_mapping = {'train': None, 'dev': None, 'test': None}
        self.x_reduce = {'train': None, 'dev': None, 'test': None}

        self.data_set_parameters = data_set_parameters
        self.scale = bool(int(data_set_parameters['scale']))
        self.scale_rows = bool(int(data_set_parameters['scale_samples']))
        self.whiten = bool(int(data_set_parameters['whiten']))
        self.pca = map(int, data_set_parameters['pca'].split())
        self.normalize_data = bool(int(data_set_parameters['normalize']))
        self.preprocessors = None

    def load(self):
        """
        Dataset can be saved as three .npy files each containing a tuple of matrices, or the data can be read manually through
        the build_dataset method. The dataset is composed of three tuples for training, testing and validation. Each
        tuple contains two matrices one for the X view and the Y view.

        The matrices are of size MxD1 and MxD2, where M is the number of samples and D1 and D2 are the dimensionality of views
        X and Y respectively.
        :return:
        """
        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        params = os.path.join(path, 'params.p')

        try:
            self.trainset = self.load_cache(path, 'train')
            self.testset = self.load_cache(path, 'test')
            self.tuning = self.load_cache(path, 'validate')

            try:
                self.x_y_mapping['test'] = numpy.load(os.path.join(path, 'mapping_test.npy'), 'r')
                self.x_y_mapping['dev'] = numpy.load(os.path.join(path, 'mapping_dev.npy'), 'r')
                self.x_reduce = cPickle.load(open(os.path.join(path, 'reduce.p'), 'r'))
            except:
                OutputLog().write('Failed loading mappings')
                self.generate_mapping()

                # Save mapping to disk
                numpy.save(os.path.join(path, 'mapping_test'), self.x_y_mapping['test'])
                numpy.save(os.path.join(path, 'mapping_dev'), self.x_y_mapping['dev'])

                with open(os.path.join(path, 'reduce.p'), 'w') as reduce_file:
                    cPickle.dump(self.x_reduce, reduce_file)

            with open(params) as params_file:
                loaded_params = cPickle.load(params_file)

            OutputLog().write('Loaded dataset params: {0}'.format(loaded_params))

        except Exception as e:
            OutputLog().write('Failed loading from local cache with exception: {}'.format(e))
            self.build_dataset()

        self.preprocess()

        OutputLog().write('Dataset dimensions = %d, %d' % (self.trainset[0].shape[1], self.trainset[1].shape[1]))
        OutputLog().write('Training set size = %d' % self.trainset[0].shape[0])
        OutputLog().write('Test set size = %d' % self.testset[0].shape[0])

        OutputLog().write('Dataset params: {0}'.format(self.data_set_parameters))

        if self.tuning is not None:
            OutputLog().write('Tuning set size = %d' % self.tuning[0].shape[0])

    def produce_optimization_sets(self, train, test_samples=None):
        """
        Creates random validation set from train set
        :param train: samples matrix
        :param test_samples: size of validation set
        :return: the new train matrix, the validation matrix and the indices of the validation set in the train set
        """
        if test_samples == 0:
            return [train, numpy.ndarray([0, 0]), 0]

        test_size = int(round(train.shape[0] / 10))

        if test_samples is None:
            test_samples = random.sample(xrange(0, train.shape[0] - 1), test_size)

        train_index = 0
        test_index = 0

        train_result = numpy.ndarray([train.shape[0] - test_size, train.shape[1]], dtype=theano.config.floatX)
        test_result = numpy.ndarray([test_size, train.shape[1]], dtype=theano.config.floatX)

        for i in xrange(train.shape[0]):

            if i in test_samples:
                test_result[test_index, :] = train[i, :]
                test_index += 1
            else:
                train_result[train_index, :] = train[i, :]
                train_index += 1

        return [train_result, test_result, test_samples]

    def preprocess(self, copy=False):
        """
        Creates a preprocessing function according to the settings, the function is used by the user of the dataset
        :param copy: Boolean specifying if there is a need to copy the input matrix
        :return:
        """
        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        if self.normalize_data:
            self.preprocessors = (functools.partial(preprocessing.normalize, copy=copy),
                                  functools.partial(preprocessing.normalize, copy=copy))

        if self.scale:
            self.preprocessors = (preprocessing.StandardScaler(copy=copy).fit(self.trainset[0]).transform,
                                  preprocessing.StandardScaler(copy=copy).fit(self.trainset[1]).transform)

        if self.scale_rows:
            self.preprocessors = (functools.partial(preprocessing.scale, copy=copy, axis=1),
                                  functools.partial(preprocessing.scale, copy=copy, axis=1))

        if not self.pca[0] == 0:
            self.preprocessors = (
                PCA(self.pca[0], copy=copy, whiten=self.whiten).fit(self.trainset[0].copy()).transform,
                lambda x: x)

        if not self.pca[1] == 0:
            self.preprocessors = (lambda x: x,
                                  PCA(self.pca[0], copy=copy, whiten=self.whiten).fit(
                                      self.trainset[1].copy()).transform)

        if self.whiten:
            OutputLog().write('using whiten')
            pca_dim1 = PCA(whiten=True)
            pca_dim2 = PCA(whiten=True)

            pca_dim1.fit(self.trainset[0])
            pca_dim2.fit(self.trainset[1])

            self.trainset = (pca_dim1.transform(self.trainset[0]), pca_dim2.transform(self.trainset[1]))
            self.testset = (pca_dim1.transform(self.testset[0]), pca_dim2.transform(self.testset[1]))
            self.tuning = (pca_dim1.transform(self.tuning[0]), pca_dim2.transform(self.tuning[1]))

    def dump(self, suffix=''):

        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        train_file = os.path.join(path, 'train{0}.p'.format(suffix))
        test_file = os.path.join(path, 'test{0}.p'.format(suffix))
        validate_file = os.path.join(path, 'validate{0}.p'.format(suffix))
        params_file = os.path.join(path, 'params{0}.p'.format(suffix))
        mapping_file = os.path.join(path, 'mapping{0}.p'.format(suffix))

        numpy.save('x_{0}'.format(train_file), self.trainset[0])
        numpy.save('y_{0}'.format(train_file), self.trainset[1])

        # hickle.dump(self.trainset, file(train_file, 'w'))
        hickle.dump(self.testset, file(test_file, 'w'))
        hickle.dump(self.tuning, file(validate_file, 'w'))
        hickle.dump(self.data_set_parameters, file(params_file, 'w'))
        hickle.dump({'map': self.x_y_mapping, 'reduce': self.x_reduce}, file(mapping_file, 'w'))

    @abc.abstractmethod
    def build_dataset(self):
        """main method for building a specific dataset"""
        return

    def generate_mapping(self):
        """
        Creates a mapping matrix from x and y, should be implemented in the child class
        :return:
        """
        pass

    def load_cache(self, path, type, mmap_type='r'):
        dataset_x = numpy.load(os.path.join(path, type + '_x.npy'), mmap_type)
        dataset_y = numpy.load(os.path.join(path, type + '_y.npy'), mmap_type)

        return (dataset_x, dataset_y)
