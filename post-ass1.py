import tensorflow as tf
sess = tf.Session()

def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


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
    c3 = tf.placeholder(tf.float32)
    af = tf.add(c1,c2)
    af = tf.add(af,c3)
    return af, c3

def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    out_value = tf.maximum(0.0,in_value)
    return out_value

def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tests here

    """

    i = tf.placeholder(shape=[x], dtype=tf.float32)
    v = tf.get_variable("v", shape=[x], initializer=tf.constant_initializer(1.0))
    mul = tf.multiply(i, v)
    iw = tf.assign(v, mul)

    ws = tf.reduce_sum(iw)
    out = my_relu(ws)

    init = tf.initialize_all_variables()
    sess.run(init)

    return out, i





#print(sess.run(add_consts()))

#af, c3 = add_consts_with_placeholder()
#print(sess.run(af, feed_dict={c3: 10}))

#print(sess.run(my_relu(-1)))

#i, ws, out = my_perceptron(5)
#print(sess.run(out, feed_dict={i: [1.2, -1.3, 1.4, 1.5, 1.6]}))

out, i = my_perceptron(4)
print(sess.run(out, feed_dict={i:[0,-1,-2,1]}))