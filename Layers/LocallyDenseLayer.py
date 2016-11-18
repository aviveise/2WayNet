import lasagne
import numpy as np

from lasagne import nonlinearities
from lasagne.layers import Layer, DenseLayer
from theano import tensor as T
from lasagne import init


class LocallyDenseLayer(Layer):
    """
    A partially fully connected layer, where the input vector of size n is separated into m different parts. Each part is
    connected into a different dense layer of size k / m, where k is the output size.
    The output of all m layers are then concatenated to create the locall dense layer output.
    As a result of this structure, the number of parameters is reduced by a factor of m comparing to regular dense layer

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape

    num_units : the output size n

    cell_num : the number of cells m

    """

    def __init__(self, incoming, num_units, cell_num, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 name=None, **kwargs):
        super(LocallyDenseLayer, self).__init__(incoming, name)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.cell_input_size = num_inputs / cell_num
        self.cell_size = self.num_units / cell_num

        if isinstance(W, lasagne.init.Initializer):
            W = [W for i in range(0, cell_num)]

        if isinstance(b, lasagne.init.Initializer):
            b = [b for i in range(0, cell_num)]

        self._dense_layers = []
        self.W = []
        self.b = []

        # Creating m number of tied dense layers
        for i in range(cell_num):
            self._dense_layers.append(TiedDenseLayer(CutLayer(incoming, cell_num),
                                                     self.cell_size, W[i], b[i], nonlinearity, **kwargs))

            self.W.append(self._dense_layers[-1].W)
            self.b.append(self._dense_layers[-1].b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        outputs = []

        # For each tied dense layer, a slice of the input is fed and a resulting n / m output is produced
        for i, layer in enumerate(self._dense_layers):
            outputs.append(layer.get_output_for(input[:, self.cell_input_size * i: self.cell_input_size * (i + 1)]))

        # The result is a concatenation of the dense layer's output
        return T.concatenate(outputs, axis=1)

    def get_params(self, **tags):
        results = []

        for layer in self._dense_layers:
            results.extend(layer.get_params(**tags))

        return results

    def __str__(self):
        return 'LocallyDenseLayer {0} units'.format(self.num_units)


class CutLayer():
    def __init__(self, layer, slice_number):
        """
        Links to a slice of the a given layer
        :param layer:
        :param slice_number:
        """
        self._layer = layer
        self.output_shape = (layer.output_shape[0], layer.output_shape[1] / slice_number)


class TiedDenseLayer(DenseLayer):
    def __init__(self, incoming, num_units,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, name=None, **kwargs):
        """
        An extention of a regular dense layer, enables the sharing of weight between two tied hidden layers. In order
        to tie two layers, the first should be initialized with an initialization function for the weights, the other
        should get the weight matrix of the first at input
        :param incoming: the input layer of this layer
        :param num_units: output size
        :param W: weight initialization, can be a initialization function or a given matrix
        :param b: bias initialization
        :param nonlinearity: non linearity function
        :param name: string
        :param kwargs:
        """
        super(TiedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, name=name)

        if not isinstance(W, lasagne.init.Initializer):
            self.params[self.W].remove('trainable')
            self.params[self.W].remove('regularizable')

        if self.b and not isinstance(b, lasagne.init.Initializer):
            self.params[self.b].remove('trainable')

    def __str__(self):
        return 'TiedDenseLayer {0} units'.format(self.num_units)
