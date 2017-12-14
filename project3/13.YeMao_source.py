# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 22:52:46 2017

@author: MAngO
"""
''' Dependencies: mxnet required, and all dependencies of mxnet like opencv etc.
    Running environment: Python 3.6
'''

import mxnet as mx
import mxnet.ndarray as nd
import numpy
import time
import ssl
import os

def load_mnist(training_num=50000):
    data_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'mnist.npz')
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))
        urllib.request.urlretrieve(origin, data_path)
        print('Done!')
    dat = numpy.load(data_path)
    X = (dat['X'][:training_num] / 126.0).astype('float32')
    Y = dat['Y'][:training_num]
    X_test = (dat['X_test'] / 126.0).astype('float32')
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def sample_test_acc(exe, X, Y, label_num=None, minibatch_size=100):
    pred = numpy.zeros((X.shape[0], label_num)).astype('float32')
    iter = mx.io.NDArrayIter(data=X, label=Y, batch_size=minibatch_size, shuffle=False)
    curr_instance = 0
    iter.reset()
    for batch in iter:
        exe.arg_dict['data'][:] = batch.data[0]
        exe.forward(is_train=False)
        batch_size = minibatch_size - batch.pad
        pred[curr_instance:curr_instance + minibatch_size - batch.pad, :] += exe.outputs[0].asnumpy()[:batch_size]
        curr_instance += batch_size
    correct = (pred.argmax(axis=1) == Y).sum()
    total = Y.shape[0]
    acc = correct/float(total)
    return correct, total, acc


def get_executor(sym, ctx, data_inputs, initializer=None):
    data_shapes = {k: v.shape for k, v in data_inputs.items()}
    arg_names = sym.list_arguments()
    aux_names = sym.list_auxiliary_states()
    param_names = list(set(arg_names) - set(data_inputs.keys()))
    arg_shapes, output_shapes, aux_shapes = sym.infer_shape(**data_shapes)
    arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
    params = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
    params_grad = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
    aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
    exe = sym.bind(ctx=ctx, args=dict(params, **data_inputs),
                   args_grad=params_grad,
                   aux_states=aux_states)
    if initializer != None:
        for k, v in params.items():
            initializer(k, v)
    return exe, params, params_grad, aux_states

def DistilledSGLD(teacher_sym, student_sym,
                  teacher_data_inputs, student_data_inputs,
                  X, Y, X_test, Y_test, total_iter_num,
                  teacher_learning_rate, student_learning_rate,
                  teacher_lr_scheduler=None, student_lr_scheduler=None,
                  student_optimizing_algorithm='adam',
                  teacher_prior_precision=1, student_prior_precision=0.001,
                  perturb_deviation=0.001,
                  student_initializer=None,
                  teacher_initializer=None,
                  minibatch_size=100,
                  dev=mx.gpu()):
    teacher_exe, teacher_params, teacher_params_grad, _ = \
        get_executor(teacher_sym, dev, teacher_data_inputs, teacher_initializer)
    student_exe, student_params, student_params_grad, _ = \
        get_executor(student_sym, dev, student_data_inputs, student_initializer)
    teacher_label_key = list(set(teacher_data_inputs.keys()) - set(['data']))[0]
    student_label_key = list(set(student_data_inputs.keys()) - set(['data']))[0]
    teacher_optimizer = mx.optimizer.create('sgld',
                                            learning_rate=teacher_learning_rate,
                                            rescale_grad=X.shape[0] / float(minibatch_size),
                                            lr_scheduler=teacher_lr_scheduler,
                                            wd=teacher_prior_precision)
    student_optimizer = mx.optimizer.create(student_optimizing_algorithm,
                                            learning_rate=student_learning_rate,
                                            rescale_grad=1.0 / float(minibatch_size),
                                            lr_scheduler=student_lr_scheduler,
                                            wd=student_prior_precision)
    teacher_updater = mx.optimizer.get_updater(teacher_optimizer)
    student_updater = mx.optimizer.get_updater(student_optimizer)
    start = time.time()
    for i in xrange(total_iter_num):
        # 1.1 Draw random minibatch
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]
        
        # 1.2 Update teacher
        teacher_exe.arg_dict['data'][:] = X_batch
        teacher_exe.arg_dict[teacher_label_key][:] = Y_batch
        teacher_exe.forward(is_train=True)
        teacher_exe.backward()       
        for k in teacher_params:
            teacher_updater(k, teacher_params_grad[k], teacher_params[k])
    
        # 2.1 Draw random minibatch and do random perturbation
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_student_batch = X[indices] + numpy.random.normal(0, perturb_deviation, X_batch.shape).astype('float32')

        # 2.2 Get teacher predictions
        teacher_exe.arg_dict['data'][:] = X_student_batch
        teacher_exe.forward(is_train=False)
        teacher_pred = teacher_exe.outputs[0]
        teacher_pred.wait_to_read()

        # 2.3 Update student
        student_exe.arg_dict['data'][:] = X_student_batch
        student_exe.arg_dict[student_label_key][:] = teacher_pred
        student_exe.forward(is_train=True)
        student_exe.backward()
        for k in student_params:
            student_updater(k, student_params_grad[k], student_params[k])

        if (i + 1) % 2000 == 0:
            end = time.time()
            print("Current Iter Num: %d" % (i + 1), "Time Spent: %f" % (end - start))
            test_correct, test_total, test_acc = \
                sample_test_acc(student_exe, X=X_test, Y=Y_test, label_num=10,
                                minibatch_size=minibatch_size)
            train_correct, train_total, train_acc = \
                sample_test_acc(student_exe, X=X, Y=Y, label_num=10,
                                minibatch_size=minibatch_size)
            teacher_test_correct, teacher_test_total, teacher_test_acc = \
                sample_test_acc(teacher_exe, X=X_test, Y=Y_test, label_num=10,
                                minibatch_size=minibatch_size)
            teacher_train_correct, teacher_train_total, teacher_train_acc = \
                sample_test_acc(teacher_exe, X=X, Y=Y, label_num=10,
                                minibatch_size=minibatch_size)
            print("Student: Test %d/%d=%f, Train %d/%d=%f" % (test_correct, test_total, test_acc,
                                                       train_correct, train_total, train_acc))
            print("Teacher: Test %d/%d=%f, Train %d/%d=%f" \
                  % (teacher_test_correct, teacher_test_total, teacher_test_acc,
                     teacher_train_correct, teacher_train_total, teacher_train_acc))
            start = time.time()

class CrossEntropySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(CrossEntropySoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1))).astype('float32')
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = (y - l)

        
class BiasXavier(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        scale = numpy.sqrt(self.magnitude / arr.shape[0])
        mx.random.uniform(-scale, scale, out=arr)
        
        
def get_mnist_sym(output_op=None, num_hidden=400):
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc1', num_hidden=num_hidden)
    net = mx.symbol.Activation(data=net, name='mnist_relu1', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc2', num_hidden=num_hidden)
    net = mx.symbol.Activation(data=net, name='mnist_relu2', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc3', num_hidden=10)
    if output_op is None:
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    else:
        net = output_op(data=net, name='softmax')
    return net

def dev():
    return mx.gpu()

def run_mnist_DistilledSGLD(training_num=50000):
    X, Y, X_test, Y_test = load_mnist(training_num)
    minibatch_size = 100
    if training_num >= 10000:
        num_hidden = 800
        total_iter_num = 1000000
        teacher_learning_rate = 1E-6
        student_learning_rate = 0.0001
        teacher_prior = 1
        student_prior = 0.1
        perturb_deviation = 0.1
    else:
        num_hidden = 400
        total_iter_num = 20000
        teacher_learning_rate = 4E-5
        student_learning_rate = 0.0001
        teacher_prior = 1
        student_prior = 0.1
        perturb_deviation = 0.001
    teacher_net = get_mnist_sym(num_hidden=num_hidden)
    crossentropy_softmax = CrossEntropySoftmax()
    student_net = get_mnist_sym(output_op=crossentropy_softmax, num_hidden=num_hidden)
    data_shape = (minibatch_size,) + X.shape[1::]
    teacher_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                           'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                           'softmax_label': nd.zeros((minibatch_size, 10), ctx=dev())}
    teacher_initializer = BiasXavier(factor_type="in", magnitude=1)
    student_initializer = BiasXavier(factor_type="in", magnitude=1)
    DistilledSGLD(teacher_sym=teacher_net, student_sym=student_net,
                  teacher_data_inputs=teacher_data_inputs,
                  student_data_inputs=student_data_inputs,
                  X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=total_iter_num,
                  student_initializer=student_initializer,
                  teacher_initializer=teacher_initializer,
                  student_optimizing_algorithm="adam",
                  teacher_learning_rate=teacher_learning_rate,
                  student_learning_rate=student_learning_rate,
                  teacher_prior_precision=teacher_prior, student_prior_precision=student_prior,
                  perturb_deviation=perturb_deviation, minibatch_size=100, dev=dev())
    
if __name__ == '__main__':
    numpy.random.seed(100)
    mx.random.seed(100)
    run_mnist_DistilledSGLD(500)