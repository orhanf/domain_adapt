import numpy as np
import theano
from theano import tensor

floatX = theano.config.floatX


def init_param(inp_size, out_size, name, scale=0.01, ortho=False):
    if ortho and inp_size == out_size:
        u, s, v = np.linalg.svd(np.random.randn(inp_size, inp_size))
        W = u.astype('float32')
    else:
        W = scale * np.random.randn(inp_size, out_size).astype(floatX)
    return theano.shared(W, name=_g(name, 'W'))


def init_bias(layer_size, name):
    return theano.shared(np.zeros(layer_size, dtype=floatX),
                         name=_g(name, 'b'))


def _p(p, q):
    return '{}_{}'.format(p, q)


class ReverseGradient(theano.gof.Op):
    __props__ = ()

    def __init__(self):
        super(ReverseGradient, self).__init__()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [-1. * output_grads[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class DenseLayer(object):
    def __init__(self, nin, dim, activ='lambda x: tensor.tanh(x)', prefix='ff',
                 scale=0.01, ortho=False, **kwargs):
        self.nin = nin
        self.dim = dim
        self.activ = activ
        self.prefix = prefix
        self.W = None
        self.b = None

    def init(self, add_bias=True):
        self.W = init_param(self.nin, self.dim, _p(self.prefix, 'W'),
                            scale=self.scale, ortho=self.ortho)
        if add_bias:
            self.b = init_bias(self.dim, _p(self.prefix, 'b'))

    def fprop(self, state_below):
        pre_act = tensor.dot(state_below, self.W) + \
            (self.b if self.b is not None else 0.)
        return eval(self.activ)(pre_act)

    def get_params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


def g_f(z, theta_f):
    for w_f, b_f in theta_f:
        z = tensor.tanh(theano.dot(z, w_f) + b_f)
    return z


def g_y(z, theta_y):
    for w_y, b_y in theta_y[:-1]:
        z = tensor.tanh(theano.dot(z, w_y) + b_y)
    w_y, b_y = theta_y[-1]
    z = tensor.nnet.softmax(theano.dot(z, w_y) + b_y)
    return z


def g_d(z, theta_d):
    for w_d, b_d in theta_d[:-1]:
        z = tensor.tanh(theano.dot(z, w_d) + b_d)
    w_d, b_d = theta_d[-1]
    z = tensor.nnet.sigmoid(theano.dot(z, w_d) + b_d)
    return z


def l_y(z, y):
    return tensor.nnet.categorical_crossentropy(z, y).mean()


def l_d(z, d):
    return tensor.nnet.binary_crossentropy(z, d).mean()


def _g(p, q):
    return '{}_{}'.format(p, q)


def mlp_parameters(input_size, layer_sizes, name=None):
    parameters = []
    previous_size = input_size
    for i, layer_size in enumerate(layer_sizes):
        parameters.append(
            (init_param(previous_size, layer_size, name=_g(name, i)),
             init_bias(layer_size, name=_g(name, i))))
        previous_size = layer_size
    return parameters, previous_size


def build_model(xs, xt, ys, input_size, hp_lambda, hp_mu,
                f_layer_sizes, y_layer_sizes, d_layer_sizes,
                simple=False):
    r = ReverseGradient(hp_lambda)

    theta_f, f_size = mlp_parameters(input_size, f_layer_sizes, 'f')
    theta_y, _ = mlp_parameters(f_size, y_layer_sizes, 'y')
    theta_d, _ = mlp_parameters(f_size, d_layer_sizes, 'd')

    if simple:
        fs = g_f(xs, theta_f)
        e = l_y(g_y(fs, theta_y), ys) + \
            l_d(g_d(r(fs), theta_d), 0) + \
            l_d(g_d(r(g_f(xt, theta_f)), theta_d), 1)
    else:
        pass

    e.name = 'cost'

    thetas = [p for theta in theta_f + theta_y + theta_d for p in theta]
    grads = [theano.grad(e, p) for p in thetas]

    updates = [(p, p - hp_mu * g) for p, g in zip(thetas, grads)]
    train = theano.function([xs, xt, ys], outputs=e, updates=updates)

    return train


def main():
    theano.config.compute_test_value = 'raise'
    xs = tensor.matrix('xs')
    xt = tensor.matrix('xt')
    ys = tensor.ivector('ys')
    hp_lambda = tensor.scalar('hp_lambda')
    hp_mu = tensor.scalar('hp_mu')
    xs.tag.test_value = np.random.randn(9, 2).astype(floatX)
    xt.tag.test_value = np.random.randn(10, 2).astype(floatX)
    ys.tag.test_value = np.random.randint(8, size=9).astype(np.int32)
    np.random.seed(4321)

    build_model(xs, xt, ys, hp_lambda, hp_mu,
                input_size=2, f_layer_sizes=[3, 4], y_layer_sizes=[7, 8],
                d_layer_sizes=[5, 6])


if __name__ == "__main__":
    main()
