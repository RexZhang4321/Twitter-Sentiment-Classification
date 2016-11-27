import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layer
from lasagne.layers import get_output_shape
import hparams
from evaluate import ConfusionMatrix
from collections import OrderedDict
import time

MAXLEN = 140
SEED = 1234


def build_model(hyparams,
                vocab,
                nclasses=2,
                batchsize=None,
                invar=None,
                maskvar=None,
                maxlen=MAXLEN):

    embedding_dim = hyparams.embedding_dim
    nhidden = hyparams.nhidden
    bidirectional = hyparams.bidirectional
    pool = hyparams.pool
    grad_clip = hyparams.grad_clip
    init = hyparams.init

    net = OrderedDict()

    V = len(vocab)
    W = lasagne.init.Normal()

    gate_params = layer.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(),
        W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )
    cell_params = layer.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(),
        W_hid=lasagne.init.Orthogonal(),
        W_cell=None,
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )

    # define model
    net['input'] = layer.InputLayer((batchsize, maxlen), input_var=invar)
    net['mask'] = layer.InputLayer((batchsize, maxlen), input_var=maskvar)
    net['emb'] = layer.EmbeddingLayer(net['input'], input_size=V, output_size=embedding_dim, W=W)
    net['fwd1'] = layer.LSTMLayer(
        net['emb'],
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=net['mask'],
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True
    )
    if bidirectional:
        net['bwd1'] = layer.LSTMLayer(
            net['emb'],
            num_units=nhidden,
            grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            mask_input=net['mask'],
            ingate=gate_params,
            forgetgate=gate_params,
            cell=cell_params,
            outgate=gate_params,
            learn_init=True,
            backwards=True
        )

        def tmean(a, b):
            agg = theano.tensor.add(a, b)
            agg /= 2.
            return agg

        net['pool'] = layer.ElemwiseMergeLayer([net['fwd1'], net['bwd1']], tmean)
    else:
        net['pool'] = layer.ConcatLayer([net['fwd1']])
    net['dropout1'] = layer.DropoutLayer(net['pool'], p=0.5)
    net['fwd2'] = layer.LSTMLayer(
        net['dropout1'],
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=net['mask'],
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True,
        only_return_final=True
    )
    net['dropout2'] = layer.DropoutLayer(net['fwd2'], p=0.6)
    net['softmax'] = layer.DenseLayer(
        net['dropout2'],
        num_units=nclasses,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    ASSUME = {net['input']: (200, 140), net['mask']: (200, 140)}
    logstr = '========== MODEL ========== \n'
    logstr += 'vocab size: %d\n' % V
    logstr += 'embedding dim: %d\n' % embedding_dim
    logstr += 'nhidden: %d\n' % nhidden
    logstr += 'pooling: %s\n' % pool
    for lname, lyr in net.items():
        logstr += '%s %s\n' % (lname, str(get_output_shape(lyr, ASSUME)))
    logstr += '=========================== \n'
    print logstr
    return net


def iterate_minibatches(inputs, targets, batchsize, rng=None, shuffle=False):
    ''' Taken from the mnist.py example of Lasagne'''
    # print inputs.shape, targets.size
    assert inputs.shape[0] == targets.size
    if shuffle:
        assert rng is not None
        indices = np.arange(inputs.shape[0])
        rng.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# train the model! return a learned model
def learn_model(hyparams, x_train, y_train, vocab):
    RNG = np.random.RandomState(SEED)
    timestamp = time.strftime('%m%d%Y_%H%M%S')

    V = len(vocab)
    n_classes = len(set(y_train))
    print "Vocab size:", V
    print "#classes:", n_classes

    # theano vars for input and output
    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    print str(hyparams)

    print "building model..."
    network = build_model(hyparams, vocab, n_classes, invar=X, maskvar=M)
    print "building model finished."

    output = lasagne.layers.get_output(network['softmax'])
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()
    params = lasagne.layers.get_all_params(network.values())

    grad_updates = lasagne.updates.adam(cost, params)

    print "compiling training functions"
    train = theano.function([X, M, y], cost,
                            updates=grad_updates,
                            allow_input_downcast=True)

    batchsize = hyparams.batchsize
    n_epochs = hyparams.nepochs
    n_batches = x_train.shape[0] / batchsize

    begin_time = time.time()

    print "start training..."
    for epoch in xrange(n_epochs):
        train_err = 0.
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(x_train, y_train, batchsize, rng=RNG, shuffle=True):
            x_mini, y_mini = batch
            train_err += train(x_mini[:, :, 0], x_mini[:, :, 0], y_mini)
            train_batches += 1
            print '[epoch %d batch %d/%d]' % (epoch, train_batches, nbatches)

        print "Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time)

    return network


def test_model(network, x_test, y_test, vocab):
    RNG = np.random.RandomState(SEED)
    timestamp = time.strftime('%m%d%Y_%H%M%S')

    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    test_output = lasagne.layers.get_output(network['softmax'], deterministic=True)
    val_cost_func = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_func = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_func = theano.function([X, M, y], [val_cost_func, val_acc_func, preds], allow_input_downcast=True)

    test_loss, test_acc, test_pred = val_func(x_test[:, :, 0], x_test[:, :, 1], y_test)
    test_eval = ConfusionMatrix(y_test, test_pred, set(y_test))

if __name__ == '__main__':
    pass
