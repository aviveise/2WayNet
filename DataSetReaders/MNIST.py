import gzip
import numpy
import os
import struct

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase
from array import array


class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte.gz'

        self.train_img_fname = 'train-images-idx3-ubyte.gz'

        self.test_images = []

        self.train_images = []

    def load_testing(self):
        ims = self.load(os.path.join(self.path, self.test_img_fname))
        self.test_images = ims

        return ims

    def load_training(self):
        ims = self.load(os.path.join(self.path, self.train_img_fname))

        self.train_images = ims

        return ims

    @classmethod
    def load(cls, path_img):
        with gzip.open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0] * rows * cols)

        for i in xrange(size):
            images[i][:] = image_data[i * rows * cols: (i + 1) * rows * cols]

        return images

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print 'Showing num:%d' % train_label[0]
        print self.display(train_img[0])
        print
        return True

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0: render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render


class MNISTDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(MNISTDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):
        mnist = MNIST(self.dataset_path)
        train_set = numpy.array(mnist.load_training())
        test_set = numpy.array(mnist.load_testing())

        train_set_x, train_set_y = self.split_dataset(train_set)
        test_set_x, test_set_y = self.split_dataset(test_set)

        x1_train_set, x1_tuning_set, test_samples = self.produce_optimization_sets(train_set_x)
        x2_train_set, x2_tuning_set, test_samples = self.produce_optimization_sets(train_set_y, test_samples)

        self.trainset = x1_train_set, x2_train_set
        self.tuning = x1_tuning_set, x2_tuning_set
        self.testset = test_set_x, test_set_y

    def split_dataset(self, dataset):
        result_a = numpy.ndarray((dataset.shape[0], dataset.shape[1] / 2))
        result_b = numpy.ndarray((dataset.shape[0], dataset.shape[1] / 2))

        for index, image in enumerate(dataset):
            image = numpy.array(image).reshape((28, 28)).reshape((784, 1), order='F')
            image_first = image[0:392]
            image_second = image[392:784]
            result_a[index, :] = image_first.flatten()
            result_b[index, :] = image_second.flatten()

        return result_a, result_b