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

	optimizer = tf.train.AdamOptimizer(1e-3)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	grads_and_vars = optimizer.compute_gradients(cross_entropy)
	train_op = optimizer.apply_gradients(grads_and_vars)
	
	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for _ in tqdm(range(NUM_EPOCHS)):
			sess.run([train_step,cross_entropy], feed_dict={x: train_x, y_: train_y,dropout_keep_prob:0.5})
	        save_path = saver.save(sess, GRAPH_SAVE_FILE)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print("Accuracy: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y,dropout_keep_prob:1.0}))

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
	# embedding layer
	with tf.device('/cpu:0'), tf.name_scope("embedding"):
		embedding_size = 128
		W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
		embedded_chars = tf.nn.embedding_lookup(W, X)
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
	# convolution + maxpool layer
	num_filters = 128
	filter_sizes = [3,4,5]
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
			conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
			h = tf.nn.relu(tf.nn.bias_add(conv, b))
			pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
			pooled_outputs.append(pooled)

	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(3, pooled_outputs)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	# dropout
	with tf.name_scope("dropout"):
		h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
	# output
	with tf.name_scope("output"):
		W = tf.get_variable("W", shape=[num_filters_total, output_size], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[output_size]))
		output = tf.nn.xw_plus_b(h_drop, W, b)
	
	y_ = tf.placeholder(tf.float32, [None, output_size], name="y_")

	return x, output, y_