import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

n_inputs = 28 * 28
n_hiddenl1 = 500
n_hiddenl2 = 500
n_classes = 10
batch_size = 100
learning_rate = 0.001
classes = { '0': 'A', '1':'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G',
                '7': 'H', '8': 'I', '9': 'J'}

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
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden2_layer['weights']), hidden2_layer['biases'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
    return output

def use_neural_network(input_img):
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, "./trained_data/trained_neural_network_mnist_A-J--e100.ckpt")
        output_nn = prediction.eval({x: [input_img]}, session = sess)
        result = np.argmax(output_nn, 1)
        letter = classes[str(result[0])]
        print("Result: ", letter)
        
img = Image.open("../media/preprocessing_data/result_bw.png")
img_array = np.array(img, dtype=np.uint8).ravel()
use_neural_network(img_array)