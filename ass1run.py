import tensorflow as tf
import numpy as np
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
    ph1 = tf.placeholder(tf.float32, shape=[], name="init")
    c_result = tf.add(c1, c2)
    result = tf.add(c_result, ph1)
    return result, ph1
    #return tf.tuple([result, ph1])

sess = tf.Session()

#sess.run(tf.global_variables_initializer())

[a, b] = add_consts_with_placeholder()

print(sess.run(a, feed_dict={b: 5.0}))

