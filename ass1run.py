import tensorflow as tf
import numpy as np

def add_consts():
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    an1 = tf.add(c1, c2)
    an2 = tf.add(an1, c3)
    return an2


def add_consts_with_placeholder():
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    ph1 = tf.placeholder(tf.float32)
    c_result = tf.add(c1, c2)
    result = tf.add(c_result, ph1)

    return result, ph1

def my_relu(in_value):
    max = tf.maximum(in_value, 0.0)
    return max

def my_perceptron(x):
    num_input = x
    num_unit = 1  # Single Perceptron

    # placeholders for input n output
    px = tf.placeholder(dtype=tf.float32, shape=[num_input])

    # Weight
    weight = tf.Variable(tf.random_uniform( [num_input, num_unit], 1.0, 1.0 ))

    # Bias
    #bias = tf.Variable(tf.zeros([num_unit]), name="Bias")

    # ActivationFunction( X * Weight + Bias)
    L = my_relu(tf.add(tf.matmul(px, weight)))

    return px, L

###############
sess = tf.Session()

size = 4
num_input = 4
mata = tf.placeholder(tf.float32, shape=[1,size])
weight = tf.placeholder(tf.float32, shape=[num_input,1])


res = tf.matmul(weight,mata)

print(sess.run(weight, feed_dict={weight:[1.0],[1.0], [1.0], [1.0] }))
print(sess.run(mata, feed_dict={mata:[4.0,3.0,2.0,1.0]}))





