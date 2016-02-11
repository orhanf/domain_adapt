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
    return theano.shared(W, name=name)


def init_bias(layer_size, name):
    return theano.shared(np.zeros(layer_size, dtype=floatX), name=name)


def _p(p, q, r):
    return '{}_{}_{}'.format(p, q, r)


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
                 postfix='0', scale=0.01, ortho=False, add_bias=True):
        self.activ = activ
        self.add_bias = add_bias
        self.W = init_param(nin, dim, _p(prefix, 'W', postfix),
                            scale=scale, ortho=ortho)
        if add_bias:
            self.b = init_bias(dim, _p(prefix, 'b', postfix))

    def fprop(self, state_below):
        pre_act = tensor.dot(state_below, self.W) + \
            (self.b if self.add_bias else 0.)
        return eval(self.activ)(pre_act)

    def get_params(self):
        return [self.W] + ([self.b] if self.add_bias else [])


class MultiLayer(object):
    def __init__(self, nin, dims, **kwargs):
        self.layers = []
        for i, dim in enumerate(dims):
            self.layers.append(DenseLayer(nin, dim, postfix=i, **kwargs))
            nin = dim

    def fprop(self, inp):
        for i, layer in enumerate(self.layers):
            inp = layer.fprop(inp)
        return inp

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


def build_model(xs, xt, ys, hp_lambda, hp_mu,
                input_dim, f_layer_dims, y_layer_dims, d_layer_dims):
    r = ReverseGradient()  # our guy

    g_f = MultiLayer(input_dim, f_layer_dims)  # feature func
    g_d = MultiLayer(f_layer_dims[-1], d_layer_dims)  # domain classifier
    g_y = MultiLayer(f_layer_dims[-1], y_layer_dims)  # label classifier

    fs = g_f.fprop(xs)
    ft = g_f.fprop(xt)

    ys_probs = tensor.nnet.softmax(g_y.fprop(fs))
    yt_prob0 = tensor.nnet.sigmoid(g_d.fprop(r(fs)))
    yt_prob1 = tensor.nnet.sigmoid(g_d.fprop(r(ft)))

    cost_ys = tensor.nnet.categorical_crossentropy(ys_probs, ys).mean()
    cost_yt = tensor.nnet.binary_crossentropy(yt_prob0, 0.).mean() + \
        tensor.nnet.binary_crossentropy(yt_prob1, 1.).mean()
    cost = cost_ys + hp_lambda * cost_yt
    cost.name = 'cost'

    params = g_f.get_params() + g_d.get_params() + g_y.get_params()

    grads = [theano.grad(cost, p) for p in params]
    updates = [(p, p - hp_mu * g) for p, g in zip(params, grads)]

    return cost, cost_ys, cost_yt, params, grads, updates


def main():
    # spawn theano vars
    xs = tensor.matrix('xs')
    xt = tensor.matrix('xt')
    ys = tensor.ivector('ys')
    hp_lambda = tensor.scalar('hp_lambda')
    hp_mu = tensor.scalar('hp_mu')

    # use test values
    batch_size = 10
    theano.config.compute_test_value = 'raise'
    xs.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    xt.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    ys.tag.test_value = np.random.randint(8, size=batch_size).astype(np.int32)
    hp_lambda.tag.test_value = 0.5
    hp_mu.tag.test_value = 1.
    np.random.seed(4321)

    # build cgs
    cost, cost_l, cost_d, param, grad, updates = build_model(
        xs, xt, ys, hp_lambda, hp_mu,
        input_dim=2, f_layer_dims=[3, 4], y_layer_dims=[7, 8],
        d_layer_dims=[5, 6])

    # compile
    train = theano.function(inputs=[xs, xt, ys, hp_mu, hp_lambda],
                            outputs=[cost, cost_l, cost_d],
                            updates=updates)

    # training loop
    niter = 1000
    ps = np.linspace(0, 1, num=niter).astype(floatX)
    learning_rate = np.float32(1.)
    gamma = 10.
    for i in range(niter):
        xs_ = np.random.randn(batch_size, 2).astype(floatX)
        xt_ = np.random.randn(batch_size, 2).astype(floatX)
        ys_ = np.random.randint(8, size=batch_size).astype(np.int32)

        lambda_p = np.float32(2. / (1. + np.exp(-gamma * ps[i])) - 1.)

        c, cl, cd = train(xs_, xt_, ys_, learning_rate, lambda_p)
        print('iter: {} - cost: {} [label: {} domain: {}] - lambda_p: {}'
              .format(i, c, cl, cd, lambda_p))

if __name__ == "__main__":
    main()
