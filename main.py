import numpy as np
import theano
import theano.tensor as tt

floatX = theano.config.floatX


class ReverseGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


def init_param(inp_size, out_size):
    return theano.shared(
        np.random.randn(inp_size, out_size).astype(floatX))


def init_bias(layer_size):
    return theano.shared(np.zeros(layer_size, dtype=floatX))


def g_f(z, theta_f):
    for w_f, b_f in theta_f:
        z = tt.tanh(theano.dot(z, w_f) + b_f)
    return z


def g_y(z, theta_y):
    for w_y, b_y in theta_y[:-1]:
        z = tt.tanh(theano.dot(z, w_y) + b_y)
    w_y, b_y = theta_y[-1]
    z = tt.nnet.softmax(theano.dot(z, w_y) + b_y)
    return z


def g_d(z, theta_d):
    for w_d, b_d in theta_d[:-1]:
        z = tt.tanh(theano.dot(z, w_d) + b_d)
    w_d, b_d = theta_d[-1]
    z = tt.nnet.sigmoid(theano.dot(z, w_d) + b_d)
    return z


def l_y(z, y):
    return tt.nnet.categorical_crossentropy(z, y).mean()


def l_d(z, d):
    return tt.nnet.binary_crossentropy(z, d).mean()


def mlp_parameters(input_size, layer_sizes):
    parameters = []
    previous_size = input_size
    for layer_size in layer_sizes:
        parameters.append(
            (init_param(previous_size, layer_size), init_bias(layer_size)))
        previous_size = layer_size
    return parameters, previous_size


def compile(input_size, f_layer_sizes, y_layer_sizes, d_layer_sizes,
            hp_lambda, hp_mu):
    r = ReverseGradient(hp_lambda)

    theta_f, f_size = mlp_parameters(input_size, f_layer_sizes)
    theta_y, _ = mlp_parameters(f_size, y_layer_sizes)
    theta_d, _ = mlp_parameters(f_size, d_layer_sizes)

    xs = tt.matrix('xs')
    xs.tag.test_value = np.random.randn(9, input_size).astype(floatX)
    xt = tt.matrix('xt')
    xt.tag.test_value = np.random.randn(10, input_size).astype(floatX)
    ys = tt.ivector('ys')
    ys.tag.test_value = np.random.randint(
        y_layer_sizes[-1], size=9).astype(np.int32)

    fs = g_f(xs, theta_f)
    e = l_y(g_y(fs, theta_y), ys) + \
        l_d(g_d(r(fs), theta_d), 0) +\
        l_d(g_d(r(g_f(xt, theta_f)), theta_d), 1)

    updates = [(p, p - hp_mu * theano.grad(e, p))
               for theta in theta_f + theta_y + theta_d for p in theta]
    train = theano.function([xs, xt, ys], outputs=e, updates=updates)

    return train


def main():
    theano.config.compute_test_value = 'raise'
    np.random.seed(1)
    compile(input_size=2, f_layer_sizes=[3, 4], y_layer_sizes=[7, 8],
            d_layer_sizes=[5, 6], hp_lambda=.5, hp_mu=.01)


main()
