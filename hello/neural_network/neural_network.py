import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

""" Mnist module
Search end extract dataset in '/tmp/data' directory with the folliwing names:
t10k-images-idx3-ubyte  - testing data
t10k-labels-idx1-ubyte  - testing labels
train-images-idx3-ubyte - train data
train-labels-idx1-ubyte - train labels
Note: replace your own mnist dataset with that names
"""
mnist = input_data.read_data_sets("./dataset")

n_inputs = 28 * 28
n_hiddenl1 = 500
n_hiddenl2 = 500
n_hiddenl3 = 500
n_hiddenl4 = 500
n_classes = 10 #[A-J]
learning_rate = 0.001
batch_size = 100

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

n_epochs = 100 #cycles feed forward + backprop
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, 'completed out of ', n_epochs, 'loss: ', epoch_loss)
            
        #save trained neural network    
        save_path = saver.save(sess, "./trained_data/trained_neural_network_mnist_A-J--e100.ckpt")
        

print("Train network...")
train_neural_network(x)
