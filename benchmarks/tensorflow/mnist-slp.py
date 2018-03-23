#!/usr/bin/env python3

import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

data_dir = os.path.join(os.getenv('HOME'), 'var/data/mnist')
mnist = input_data.read_data_sets(data_dir, one_hot=True)

def slp(imput_size, class_number):
    x = tf.placeholder(tf.float32, [None, imput_size])
    W = tf.Variable(tf.zeros([imput_size, class_number]))
    b = tf.Variable(tf.zeros([class_number]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return x, y

learning_rate = 0.5
x, y = slp(28 * 28, 10)
y_ = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optmizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optmizer.minimize(loss)

# Train
total = 60000
batch_size = 10000
n = total // batch_size

# sess = tf.InteractiveSession()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, total // batch_size):
        print('step: %d' % i)
        xs, y_s = mnist.train.next_batch(batch_size)
        train_step.run({x: xs, y_: y_s})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(acc, {x: mnist.test.images, y_: mnist.test.labels})
    print('accuracy: %f' % result)
