import numpy
import theano

from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import Layer

from theano import tensor as T

class BatchNormalizationLayer(Layer):
    """
    Batch normalization Layer [1]
    The user is required to setup updates for the learned parameters (Gamma
    and Beta). The values nessesary for creating the updates can be
    obtained by passing a dict as the moving_avg_hooks keyword to
    get_output().

    REF:
     [1] http://arxiv.org/abs/1502.03167

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - nonlinearity : callable or None (default: lasagne.nonlinearities.rectify)
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

        - epsilon : scalar float. Stabilizing training. Setting this too
            close to zero will result in nans.
    """

    def __init__(self, incoming,
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon=0.001,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = int(numpy.prod(self.input_shape[1:]))
        self.gamma = self.add_param(gamma, (self.num_units,), name="BatchNormalizationLayer:gamma", regularizable=True,
                                    gamma=True, trainable=True)
        self.beta = self.add_param(beta, (self.num_units,), name="BatchNormalizationLayer:beta", regularizable=False)
        self.epsilon = epsilon

        self.mean_inference = theano.shared(
            numpy.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.mean_inference.name = "shared:mean"

        self.variance_inference = theano.shared(
            numpy.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.variance_inference.name = "shared:variance"

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, moving_avg_hooks=None,
                       deterministic=False, *args, **kwargs):

        if deterministic is False:
            m = T.mean(input, axis=0, keepdims=True, dtype=theano.config.floatX)
            v = T.sqrt(T.var(input, axis=0, keepdims=True) + self.epsilon)
            m.name = "tensor:mean"
            v.name = "tensor:variance"

            key = "BatchNormalizationLayer:movingavg"
            if key not in moving_avg_hooks:
                moving_avg_hooks[key] = []
            moving_avg_hooks[key].append(
                [[m, v], [self.mean_inference, self.variance_inference]])
        else:
            m = self.mean_inference
            v = self.variance_inference

        input_hat = (input - m) / v  # normalize
        y = input_hat * self.gamma + self.beta  # scale and shift

        return self.nonlinearity(y)
