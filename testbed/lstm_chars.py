# -*- coding: utf-8 -*-
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
import preprocess_char
import pickle
import os

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

    # ### sanity check ###
    pad_char = u'♥'
    vocab[pad_char] = 0

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

    test_output = lasagne.layers.get_output(network['softmax'], deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_fn = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds], allow_input_downcast=True)

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
            try:
                train_err += train(x_mini[:, :, 0], x_mini[:, :, 1], y_mini)
            except:
                print x_mini
                print y_mini
                print len(x_mini), len(y_mini)
                exit(1)
            # train_err += train(x_mini[:, :, 0], x_mini[:, :, 1], y_mini)
            train_batches += 1
            print '[epoch %d batch %d/%d]' % (epoch, train_batches, n_batches)

        print "Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, n_epochs, time.time() - start_time)

    test_loss, test_acc, _ = val_fn(x_train[:, :, 0], x_train[:, :, 1], y_train)
    print test_loss, test_acc
    return network


def test_model(model_fname, x_test, y_test, hyparams, vocab):
    RNG = np.random.RandomState(SEED)
    timestamp = time.strftime('%m%d%Y_%H%M%S')

    pad_char = u'♥'
    vocab[pad_char] = 0
    n_classes = len(set(y_test))

    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    print "building model..."
    clf = build_model(hyparams, vocab, n_classes, invar=X, maskvar=M)
    read_model_from_file(clf, model_fname)
    # params = lasagne.layers.get_all_param_values(network.values())
    # lasagne.layers.set_all_param_values(clf.values(), params)
    print "model built."

    test_output = lasagne.layers.get_output(clf['softmax'], deterministic=True)
    val_cost_func = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_func = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_func = theano.function([X, M, y], [val_cost_func, val_acc_func, preds], allow_input_downcast=True)

    test_loss, test_acc, test_pred = val_func(x_test[:, :, 0], x_test[:, :, 1], y_test)
    print test_loss, test_acc
    while True:
        txt = raw_input("Type a tweet: ")
        txt = preprocess_char.load_from_one_text(txt, vocab)
        _, _, test_pred = val_func(txt[:, :, 0], txt[:, :, 1], [0])
        print "Prediction: ", test_pred


def write_model_to_file(model, fname):
    data = lasagne.layers.get_all_param_values(model.values())
    fname = os.path.join('../model', fname)
    with open(fname, 'w+') as f:
        pickle.dump(data, f)


def read_model_from_file(model, fname):
    path = os.path.join('../model', fname)
    with open(path, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model.values(), data)

if __name__ == '__main__':
    path = '../data/training2.csv'
    names = ["classes", "id", "time", "query", "user", "data"]
    usecols = [0, 5]
    # path = '../data/semeval/train.tsv'
    # names = ["id", "classes", "data"]
    # usecols = [1, 2]
    print "loading data..."
    x_train, y_train, vocab = preprocess_char.load_from_file(path, names=names, usecols=usecols)
    print y_train
    hyparams = hparams.HParams()
    hyparams.bidirectional = False
    print hyparams
    x_train = x_train[:4000]
    y_train = y_train[:4000]
    cnt = 0
    for i in y_train:
        if i == 0:
            cnt += 1
    print cnt
    clf = learn_model(hyparams, x_train, y_train, vocab)
    fname = "test1_2point_4000_onedire"
    write_model_to_file(clf, fname)
    test_model(fname, x_train, y_train, hyparams, vocab)
