import lasagne
from math import floor

from lasagne.regularization import l2
from theano import tensor as T

from Layers.BNLayer import BatchNormalizationLayer
from params import Params


def transpose_recursive(w):
    if not isinstance(w, list):
        return w.T

    return [transpose_recursive(item) for item in w]


def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes,
                weight_init=lasagne.init.GlorotUniform()):
    """
    Creates at bi-directional model, containing two channels from var_x to the reconstruction of var_y and vice versa,
    the returned value contains also a the composite loss term.
    The loss term is composed of:

    1. The reconstruction loss between X and X' and Y and Y' (X' and Y' are the output of each channel)

    2. The reconstruction loss of the OUTPUT_LAYER from both channels

    3. The covariance regularization which aims to decorralted each output internally

    4. The gamma regularization, equals to the the sum of the squared norm of 1/gamma (the batch normalization parameter)

    5. Weight decay

    :param var_x: theano variable for the input x view
    :param input_size_x: size of x dimensionality
    :param var_y: theano variable for the input y view
    :param input_size_y: size of y dimensionality
    :param layer_sizes: array containing the sizes of hidden layers
    :param weight_init: initialization function for the weights
    :return:
    """
    layer_types = Params.LAYER_TYPES

    # Create x to y network
    model_x, hidden_x, weights_x, biases_x, prediction_y, hooks_x, dropouts_x = build_single_channel(var_x,
                                                                                                     input_size_x,
                                                                                                     input_size_y,
                                                                                                     layer_sizes,
                                                                                                     layer_types,
                                                                                                     weight_init,
                                                                                                     lasagne.init.Constant(
                                                                                                         0.),
                                                                                                     'x')

    weights_y = [transpose_recursive(w) for w in reversed(weights_x)]
    bias_y = lasagne.init.Constant(0.)

    model_y, hidden_y, weights_y, biases_y, prediction_x, hooks_y, dropouts_y = build_single_channel(var_y,
                                                                                                     input_size_y,
                                                                                                     input_size_x,
                                                                                                     list(reversed(
                                                                                                         layer_sizes)),
                                                                                                     list(reversed(
                                                                                                         layer_types)),
                                                                                                     weights_y,
                                                                                                     bias_y,
                                                                                                     'y',
                                                                                                     dropouts_x)

    reversed_hidden_y = list(reversed(hidden_y))

    hooks = {}
    if "BatchNormalizationLayer:movingavg" in hooks_x:
        # Merge the two dictionaries
        hooks = hooks_x
        hooks["BatchNormalizationLayer:movingavg"].extend(hooks_y["BatchNormalizationLayer:movingavg"])
        # hooks["WhiteningLayer:movingavg"].extend(hooks_y["WhiteningLayer:movingavg"])

    loss_x = Params.LOSS_X * lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=1).mean()
    loss_y = Params.LOSS_Y * lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=1).mean()

    hooks_temp = {}

    layer_x = lasagne.layers.get_output(hidden_x[Params.OUTPUT_LAYER], moving_avg_hooks=hooks_temp)
    layer_y = lasagne.layers.get_output(reversed_hidden_y[Params.OUTPUT_LAYER], moving_avg_hooks=hooks_temp)

    loss_l2 = Params.L2_LOSS * lasagne.objectives.squared_error(layer_x, layer_y).sum(axis=1).mean()

    loss_weight_decay = 0

    cov_x = T.dot(layer_x.T, layer_x) / T.cast(layer_x.shape[0], dtype=T.config.floatX)
    cov_y = T.dot(layer_y.T, layer_y) / T.cast(layer_x.shape[0], dtype=T.config.floatX)

    loss_withen_x = Params.WITHEN_REG_X * (T.sqrt(T.sum(T.sum(cov_x ** 2))) - T.sqrt(T.sum(T.diag(cov_x) ** 2)))
    loss_withen_y = Params.WITHEN_REG_Y * (T.sqrt(T.sum(T.sum(cov_y ** 2))) - T.sqrt(T.sum(T.diag(cov_y) ** 2)))

    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_x,
                                                                        penalty=l2) * Params.WEIGHT_DECAY
    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_y,
                                                                        penalty=l2) * Params.WEIGHT_DECAY

    gamma_x = lasagne.layers.get_all_params(model_x, gamma=True)
    gamma_y = lasagne.layers.get_all_params(model_y, gamma=True)

    loss_gamma = T.constant(0)
    loss_gamma += sum(l2(1 / gamma) for gamma in gamma_x) * Params.GAMMA_COEF
    loss_gamma += sum(l2(1 / gamma) for gamma in gamma_y) * Params.GAMMA_COEF

    loss = loss_x + loss_y + loss_l2 + loss_weight_decay + loss_withen_x + loss_withen_y + loss_gamma

    output = {
        'loss_x': loss_x,
        'loss_y': loss_y,
        'loss_l2': loss_l2,
        'loss_weight_decay': loss_weight_decay,
        'loss_gamma': loss_gamma,
        'loss_withen_x': loss_withen_x,
        'loss_withen_y': loss_withen_y,
        'mean_x': T.mean(T.mean(layer_x, axis=0)),
        'mean_y': T.mean(T.mean(layer_y, axis=0)),
        'var_x': T.mean(T.var(layer_x, axis=0)),
        'var_y': T.mean(T.var(layer_y, axis=0)),
        'var_mean_x': T.var(T.mean(layer_x, axis=0)),
        'var_mean_y': T.var(T.mean(layer_y, axis=0))
    }

    return model_x, model_y, hidden_x, reversed_hidden_y, loss, output, hooks

def build_single_channel(var, input_size, output_size, layer_sizes, layer_types,
                         weight_init=lasagne.init.GlorotUniform(),
                         bias_init=lasagne.init.Constant(0.), name='', dropouts_init=None):
    """
    Build a single channel containing layers of sizes according to layer_sizes array with initialization given
    :param var: tensor variable for the input
    :param input_size: input dimensionality
    :param output_size: ouput dimensionality
    :param layer_sizes: array of layer sizes
    :param layer_types: array of layer types
    :param weight_init: initialization function for the weights
    :param bias_init: initialization function for the biases
    :param name: name of the network
    :param dropouts_init: initialization of the tied dropout, if none samples a new drop matrix
    :return: the model containing the channels with it's weights, biases, drop matrices and batch normalization hooks
    """
    model = []
    weights = []
    biases = []
    hidden = []
    dropouts = []
    hooks = {}

    if isinstance(weight_init, lasagne.init.Initializer):
        weight_init = [weight_init for i in range(len(layer_sizes) + 1)]

    if isinstance(bias_init, lasagne.init.Initializer):
        bias_init = [bias_init for i in range(len(layer_sizes) + 1)]

    if dropouts_init is None:
        dropouts_init = [dropouts_init for i in range(len(layer_sizes) + 1)]

    # Add Input Layer
    model.append(lasagne.layers.InputLayer((None, input_size), var, 'input_layer_{0}'.format(name)))

    # Add hidden layers
    for index, layer_size in enumerate(layer_sizes):
        model.append(layer_types[index](incoming=model[-1],
                                        num_units=layer_size,
                                        W=weight_init[index],
                                        b=bias_init[index],
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(
                                            Params.LEAKINESS) if not Params.BN_ACTIVATION else lasagne.nonlinearities.identity,
                                        cell_num=Params.LOCALLY_DENSE_M))

        weights.append(model[-1].W)
        biases.append(model[-1].b)

        if Params.BN:
            model.append(BatchNormalizationLayer(model[-1],
                                                 nonlinearity=lasagne.nonlinearities.LeakyRectify(
                                                     Params.LEAKINESS) if Params.BN_ACTIVATION else lasagne.nonlinearities.identity))

        model.append(
            Params.NOISE_LAYER(model[-1], p=Params.DROP_PROBABILITY, noise_layer=dropouts_init[-(index + 1)]))

        dropouts.append(model[-1])

        hidden.append(model[-1])

    # Add output layer
    model.append(layer_types[-1](model[-1],
                                 num_units=output_size,
                                 W=weight_init[-1],
                                 b=bias_init[-1],
                                 nonlinearity=lasagne.nonlinearities.identity,
                                 cell_num=Params.LOCALLY_DENSE_M))
    weights.append(model[-1].W)
    biases.append(model[-1].b)

    prediction = lasagne.layers.get_output(model[-1], moving_avg_hooks=hooks)

    return model, hidden, weights, biases, prediction, hooks, dropouts
