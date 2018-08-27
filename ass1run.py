import tensorflow as tf
import numpy as np

def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    an1 = tf.add(c1, c2)
    an2 = tf.add(an1, c3)
    return an2


def add_consts_with_placeholder():
    """
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    ph1 = tf.placeholder(tf.float32)
    c_result = tf.add(c1, c2)
    result = tf.add(c_result, ph1)

    return result, ph1

def my_relu(in_value):
    #in_value is the output of the nueron which my_relu decieds whether it has to propogate forward or not
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    max = tf.maximum(in_value, 0.0)
    return max

###############
sess = tf.Session()

size = 4
num_input = 4
test = tf.placeholder(tf.float32, shape=[size])
mata = tf.placeholder(tf.float32, shape=[num_input])

res = tf.matmul(test,mata)

print(sess.run(test, feed_dict={test:[1.0,2.0,3.0,4.0]}))
print(sess.run(mata, feed_dict={mata:[4.0,3.0,2.0,1.0]}))





