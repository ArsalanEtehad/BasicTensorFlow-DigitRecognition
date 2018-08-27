import tensorflow as tf


def my_relu(in_value):
    # in_value is the output of the nueron which my_relu decieds whether it has to propogate forward or not
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    out_value = tf.maximum(0, in_value)
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
    num_input = len(x)
    num_unit = 1  # Single Perceptron
    num_output = 1

    # placeholders for input n output
    x = tf.placeholder(dtype=tf.float32, shape=[None, num_input])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_output])

    # Weight and Bias
    weight = tf.Variable(tf.get_variable([num_input, num_unit]),-1.0 , 1.0)
    bias = tf.Variable(tf.random_normal([num_unit]), -1.0 , 1.0)

    # ActivationFunction( X * Weight + Bias)
    output = my_relu(tf.add(tf.matmul(x, weight), bias))

    return output


x = [1.0,2.0,3.0,4.0]

num_input = len(x)
num_unit = 1  # Single Perceptron
num_output = 1

# placeholders for input n output
X = tf.placeholder(dtype=tf.float32, shape=[None, num_input])

# Weight and Bias
weight = tf.Variable(tf.random_normal([num_input, num_unit]))
bias = tf.Variable(tf.random_normal([num_unit]))

# ActivationFunction( X * Weight + Bias)
output = my_relu(tf.matmul(X, weight) + bias)