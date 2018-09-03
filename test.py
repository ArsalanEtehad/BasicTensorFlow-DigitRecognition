import tensorflow as tf

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.Session()

print(sess.run(y, feed_dict={y:[[0,0,0,0,0,0,0,0,0,1]]}))

