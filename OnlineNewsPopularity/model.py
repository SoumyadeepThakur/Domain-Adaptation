import tensorflow as tf
import numpy as np

def model(input_tensor, **kwargs):


	with tf.variable_scope('deep_cnn_layer', reuse=tf.AUTO_REUSE):

		
		feature = tf.compat.v1.layers.dense(input_tensor, 64, activation=tf.nn.relu, name='fc-1')
		feature = tf.compat.v1.layers.dense(feature, 32, activation=tf.nn.relu, name='fc-2')
		
		output_tensor = tf.compat.v1.layers.dense(feature, 1, name='fc-3')
		
	return output_tensor

def loss(y_true, y_pred):

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
	return loss






	