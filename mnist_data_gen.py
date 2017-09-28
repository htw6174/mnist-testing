import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf
import pgm_to_tf

def next_labels_batch(size):
    labels = np.zeros([size, 10])
    for i in range(size):
        value = random.randint(0, 9)
        labels[i, value] = 1.0
    return labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1]):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = strides, padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def label_images(x, sess):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.Variable(1.0, tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver()
    #correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver.restore(sess, "./models/mnist.ckpt")

    return y_conv

def test_accuracy(x, y_):
    y_conv = label_images(x)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def decode(labels):
    #labels is a tensor of shape [batch, 10]
    batch_size = tf.shape(labels)[0] #get the batch size of labels, because conv2d_transpose needs exact output dimensions

    deconv1 = tf.reshape(labels, [-1, 1, 1, 10]) #reshape to [batch, 1, 1, 10]
    W_deconv1 = weight_variable([3, 3, 8, 10])
    shape1 = tf.stack([batch_size, 2, 2, 8]) #pack shape so it can be used by conv2d_transpose

    deconv2 = conv2d_transpose(deconv1, W_deconv1, shape1, strides = [1, 2, 2, 1]) #reshape to [batch, 2, 2, 8]
    W_deconv2 = weight_variable([5, 5, 8, 8])
    shape2 = tf.stack([batch_size, 4, 4, 8])

    deconv3 = conv2d_transpose(deconv2, W_deconv2, shape2, strides = [1, 2, 2, 1]) #reshape to [batch, 4, 4, 8]
    W_deconv3 = weight_variable([5, 5, 16, 8])
    shape3 = tf.stack([batch_size, 7, 7, 16])

    deconv4 = conv2d_transpose(deconv3, W_deconv3, shape3, strides = [1, 2, 2, 1]) #reshape to [batch, 7, 7, 16]
    W_deconv4 = weight_variable([5, 5, 32, 16])
    shape4 = tf.stack([batch_size, 14, 14, 32])

    deconv5 = conv2d_transpose(deconv4, W_deconv4, shape4, strides = [1, 2, 2, 1]) #reshape to [batch, 14, 14, 32]
    W_deconv5 = weight_variable([5, 5, 1, 32])
    shape5 = tf.stack([batch_size, 28, 28, 1])

    deconv6 = conv2d_transpose(deconv5, W_deconv5, shape5, strides = [1, 2, 2, 1]) #reshape to [batch, 28, 28, 1]
    images = tf.reshape(deconv6, [-1, 784]) #reshape to [batch, 784]
    return tf.nn.relu(images) #should be applied at every step of the decode?

if __name__ == "__main__":

    #x = tf.placeholder(tf.float32, shape=[None, 10]) #input is an array of zeros with a 1 at index n, where n is the number to be drawn
    #y_ = tf.placeholder(tf.float32, shape=[None, 784]) #output is an array of intensities representing a 28x28 image

    #tf.reset_default_graph()
    #x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    #sess = tf.InteractiveSession() #slower than using tf.session
    sess = tf.Session()

    #feed_dict = pgm_to_tf.fill_feed_dict("8.pgm", x, y_)
    #accuracy = test_accuracy(x, y_)
    #accuracy = accuracy.eval(feed_dict = feed_dict, session = sess)
    #print("test accuracy {}".format(accuracy))

    x = decode(y_)
    y_classify = label_images(x, sess)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_classify))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_classify, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        feed_dict = {y_: next_labels_batch(50)}
        train_step.run(feed_dict = feed_dict)

    print("test accuracy {}".format(accuracy.eval(feed_dict = feed_dict)))
    feed_dict = {y_: next_labels_batch(50)}
    generated_images = x.eval(feed_dict = feed_dict, session = sess)
    pgm_to_tf.make_images(generated_images, feed_dict[y_])

    sess.close()
