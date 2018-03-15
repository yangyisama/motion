import os
import random
import pickle
import numpy as np
import tensorflow as tf
import midi_manipulation, dataset_manipulation

from tqdm import tqdm

GRAPH_SAVE_FILE = "./_saved_graph/model.ckpt"

### Hyperparameters
NUM_EPOCHS = 500

NUM_VISIBLE = dataset_manipulation.SONG_SLICE_COUNT
NUM_HIDDEN = 1560
NUM_CLASSES = 5

MIDI_PICKLE = "_moodset_cache/midi.pickle"

learning_rate = tf.constant(0.005, tf.float32)
weights = {'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases = {'out':tf.Variable(tf.random_normal([n_classes]))}

""" Trains songs for given labels

Feeds songs and labels to computation graph
"""
def train_songs(force_reload=False):
	if force_reload or not os.path.isfile(MIDI_PICKLE):
		train_x, train_y, test_x, test_y = dataset_manipulation.generate_moodset_pickle(MIDI_PICKLE)
	else:
		with open(MIDI_PICKLE,"r") as file:
			train_x, train_y, test_x, test_y = pickle.load(file)

	x, y, y_ = build_computation_graph()

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

	saver = tf.train.Saver()
    loss_list=[]
    epoch_loss=0
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for _ in tqdm(range(NUM_EPOCHS)):
			_, c=sess.run([cross_entropy,train_step], feed_dict={x: train_x, y_: train_y})
	        save_path = saver.save(sess, GRAPH_SAVE_FILE)
	        epoch_loss += c
	        loss_list.append(c)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print("Accuracy: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

		tf.summary.FileWriter("./_saved_graph/log", sess.graph)

""" Predicts Song

Predicts song from saved graph file
"""
def predict_song(song):
	with tf.Graph().as_default():
		x, y, y_ = build_computation_graph()

		saver = tf.train.Saver()

		with tf.Session() as sess:
			save_path = saver.restore(sess, GRAPH_SAVE_FILE)
			prediction = tf.argmax(y,1)
			return sess.run(prediction, feed_dict={x: [song]})

""" Builds Computation graph

For given input output sizes builds a 3-layer computation graph
"""
def build_computation_graph(input_size=NUM_VISIBLE, output_size=NUM_CLASSES):
    
    x  = tf.placeholder(tf.float32, [None, input_size], name="x")
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*n_layers)
    outputs,states = tf.nn.dynamic_rnn(multi_cell,x,time_major=True)
	last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
	y_ = tf.placeholder(tf.float32, [None, output_size], name="y_")

	return x, last, y_