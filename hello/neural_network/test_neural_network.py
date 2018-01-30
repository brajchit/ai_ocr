import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./dataset")

n_inputs = 28 * 28
n_hiddenl1 = 500
n_hiddenl2 = 500
n_hiddenl3 = 500
n_hiddenl4 = 500
n_classes = 10
batch_size = 100
learning_rate = 0.001

x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neural_network_model(data):
    hidden1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_hiddenl1])),
                        'biases': tf.Variable(tf.random_normal([n_hiddenl1]))}
    hidden2_layer = {'weights': tf.Variable(tf.random_normal([n_hiddenl1, n_hiddenl2])),
                        'biases': tf.Variable(tf.random_normal([n_hiddenl2]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hiddenl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden1_layer['weights']), hidden1_layer['biases'])
    l1 = tf.nn.relu(l1) #activation function dor layer 1
    l2 = tf.add(tf.matmul(l1, hidden2_layer['weights']), hidden2_layer['biases'])
    l2 = tf.nn.relu(l2) #activation function dor layer 2
    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
    return output


def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, "./trained_data/trained_neural_network_mnist_A-J:e100.ckpt")
        correct = tf.nn.in_top_k(prediction, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
print("Testing network...")
test_neural_network()