import theano

from math import sqrt
from lasagne.layers import Layer
from lasagne.random import get_rng
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams


class TiedDropoutLayer(Layer):
    """Tied Dropout layer

    Extension of the conventional dropout layer which shares the zeroing matrix which is sampled from a binomial
    distribution. The sharing is between two tied dropout layer of two tied hidden layers.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero

    noise_layer: the tied dropout layer, if none samples a new masking matrix

    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.

    """

    def __init__(self, incoming, p=0.5, noise_layer=None, **kwargs):
        super(TiedDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self._master = noise_layer
        self._mask = None

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input

        else:
            retain_prob = 1 - self.p
            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            # If connected to a nother tied dropout layer take his mask
            if self._master is not None:
                self._mask = self._master._mask

            # Otherwise sample a new mask from binomial distribution
            if self._mask is None:
                self._mask = tensor.cast(self._srng.binomial(input_shape, p=retain_prob),
                                         dtype=theano.config.floatX)

            input *= self._mask

            # The training output is scaled by a factor of sqrt(1-p) to maintain the same variance after the drop
            input /= sqrt(retain_prob)

            return input


class DropoutLayer(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, p=0.5, **kwargs):
        super(DropoutLayer, self).__init__(incoming)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            input = input * self._srng.binomial(input_shape, p=retain_prob,
                                                dtype=theano.config.floatX)

            input /= retain_prob

            return input
