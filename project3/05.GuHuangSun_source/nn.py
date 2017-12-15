import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Network parameters
batch_size = 6
n_epochs = 500
n_hidden = 6
n_classes = 2
activation = 'leaky'
optimizer = 'adam'
dropout = None
n_feat = 54
n_exp = 10
n_cv = 10

lr = 0.05
stop_threshold = 0.01
fix_batch = True
# Decaying earning rate
#global_step = tf.Variable(0, trainable=False)
#start_lr = 0.0002
#lr = tf.train.exponential_decay(start_lr, global_step, 500, 0.98, staircase=True)


# Utility functions
def get_batches(data, batch_size=batch_size, fixed=True):
    if fixed:
        np.random.shuffle(data)
        return [(data[:,:-1], data[:,-1])]
    else:
        batches = []
        x = data
        n = x.shape[0]
        np.random.shuffle(x)
        while True:
            if len(x) <= batch_size:
                inputs_batch = x[:,:-1]
                labels_batch = np.array(x[:,-1], dtype=int)
                batches.append((inputs_batch, labels_batch))
                break
            else:
                inputs_batch = x[:batch_size,:-1]
                labels_batch = np.array(x[:batch_size,-1], dtype=int)
                batches.append((inputs_batch, labels_batch))
                x = x[batch_size:,:]
        return batches

def leaky(state, alpha=0.01):
    return tf.nn.relu(state) - alpha * tf.nn.relu(-state)

# The network
# tf.contrib.layers.xavier_initializer()
inputsholder = tf.placeholder(shape=(None, n_feat), dtype=tf.float32)
labelsholder = tf.placeholder(shape=(None,), dtype=tf.int32)
w1 = tf.get_variable('w1', shape=(n_feat, n_hidden),
    initializer=tf.truncated_normal_initializer())
b1 = tf.get_variable('b1', shape=(n_hidden,),
    initializer=tf.truncated_normal_initializer())
w2 = tf.get_variable('w2', shape=(n_hidden, n_classes), \
    initializer=tf.truncated_normal_initializer())
b2 = tf.get_variable('b2', shape=(n_classes,), \
    initializer=tf.truncated_normal_initializer())

if activation == 'leaky':
    h = leaky(tf.matmul(inputsholder, w1) + b1)
else:
    exec('h = tf.nn.{}(tf.matmul(inputsholder, w1) + b1)'.format(activation))

if dropout is not None:
    h = tf.nn.dropout(h, 1 - dropout)

preds = tf.matmul(h, w2) + b2
prediction = tf.argmax(preds, axis=1)
re = tf.reduce_mean(tf.cast(tf.equal(
    tf.cast(prediction, tf.int32), labelsholder), tf.float32))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labelsholder, logits=preds))
if optimizer == 'adam':
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
else:
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(
        loss, global_step=global_step)

def evaluate(sess, data_test):
    feed = {inputsholder: data_test[:,:-1], labelsholder: data_test[:,-1]}
    accuracy = sess.run(re, feed_dict=feed)
    return accuracy

def predict(sess, data_test):
    feed = {inputsholder: data_test[:,:-1]}
    return sess.run(prediction, feed_dict=feed)

def train(sess, data_train, n_epochs, loss_threshold=stop_threshold):
    #training_loss = np.zeros(n_epochs)
    #accuracy = np.zeros(n_epochs)
    if loss_threshold is None:
        for j in range(n_epochs):
            batches = get_batches(data_train, fixed=fix_batch)
            while len(batches) > 0:
                batch = batches.pop()
                feed = {inputsholder: batch[0], labelsholder: batch[1]}
                _, l = sess.run([train_op, loss], feed_dict=feed)
            a = evaluate(sess, data_train)
            #training_loss[j] = l
            #accuracy[j] = a
            if j < n_epochs-1:
                print('Epoch: {} Loss: {:f} Accuracy: {:f}'.format(j+1, l, a), end='\r')
            else:
                print('Epoch: {} Loss: {:f} Accuracy: {:f}'.format(j+1, l, a))
        #return training_loss, accuracy
    else:
        j = 0
        l = 1
        a = 0
        while l > loss_threshold:
            j += 1
            batches = get_batches(data_train, fixed=fix_batch)
            while len(batches) > 0:
                batch = batches.pop()
                feed = {inputsholder: batch[0], labelsholder: batch[1]}
                _, l = sess.run([train_op, loss], feed_dict=feed)
            a = evaluate(sess, data_train)
            #training_loss[j] = l
            #accuracy[j] = a
            if l > loss_threshold:
                print('Epoch: {} Loss: {:f} Accuracy: {:f}'.format(j+1, l, a), end='\r')
            else:
                print('Epoch: {} Loss: {:f} Accuracy: {:f}'.format(j+1, l, a))

if __name__=='__main__':
    disputed_id = np.array([1, 7, 10, 20, 23, 25, 26]) - 1 # The IDs of the disputed paintings
    r = np.asarray([2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27, 28]) - 1 # IDs of Raphael's paintings
    nr = np.array([11, 12 ,13, 14, 15, 16, 17, 18, 19]) - 1#, \
        #29, 30, 31, 32, 33, 34, 35]) - 1 # IDs of non-Raphael paintings
    confirmed = np.sort(np.append(r, nr)) # IDs of paintings which are not disputed

    features = np.loadtxt('features.txt')[:28,:]
    n_files = features.shape[0]
    x_raw = np.zeros((n_files, n_feat+1))
    x_raw[:,:-1] = features
    x_raw[r,-1] = 1
    data = x_raw[confirmed,:]
    n_samples = len(data)

    # Cross validation
    cv_all = []
    for i in range(n_cv):
        results = []
        positive = []
        negative = []
        for i in range(n_samples):
            print('Cross Validaiton Iteration: {}'.format(i+1))
            t0 = time()
            data_test = data[[i],:]
            data_train = data[[j for j in range(n_samples) if j != i],:]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train(sess, data_train, n_epochs)
                result = evaluate(sess, data_test)
                results.append(result)
                print('Test result: {} Time used: {:f} seconds\n'.format(result, time()-t0))
            if data_test[0,-1] == 1:
                positive.append(result)
            else:
                negative.append(result)
        cv_accuracy = np.mean(results)
        cv_all.append(cv_accuracy)
        print('Cross validation accuracy: {:f}'.format(cv_accuracy))
    print('All accuracies: ', cv_all)
    print('Average accuracy: ', np.mean(cv_all))
    print('Positives: ', positive)
    print('TPR: ', np.mean(positive))
    print('Negatives: ', negative)
    print('TNR: ', np.mean(negative))

    # Make predictions on the disputed ones
    data_test = x_raw[disputed_id,:]
    data_train = x_raw[confirmed,:]
    predictions = np.zeros((n_exp, len(disputed_id)))
    for e in range(n_exp):
        print('Experiment: ', e+1)
        t0 = time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train(sess, data_train, n_epochs)
            pr = predict(sess, data_test)
            predictions[e,:] = pr
            print('Prediction: {} Time used: {:f} seconds\n'.format(pr, time()-t0))
    final_score = predictions.mean(axis=0)
    print('Average predictions: {}'.format(final_score))
