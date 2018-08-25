import tensorflow as tf

def add_consts_with_placeholder(data):
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
    add_cons = tf.add(c1, c2)
    af = tf.add(add_cons, ph1)

    return tf.tuple([af,ph1])



def my_relu(in_value):
    #in_value is the output of the nueron which my_relu decieds whether it has to propogate forward or not
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    out_value = tf.maximum(0,in_value)
    return out_value


sess = tf.Session()
#ppp = sess.run(add_consts_with_placeholder(),feed_dict={add_consts_with_placeholder():4})
i = tf.placeholder(tf.float32)
output = my_relu(sess.run(i,feed_dict={i:15.1}))
print(sess.run(output))