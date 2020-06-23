import numpy as np
import tensorflow as tf

class NN_Model():

	def __init__(self, input_shape):

		self.input_shape = input_shape
		self.model_ = None
		self.disc_ = None
		self.loss_ = None

	def create_model(self, input_tensor, source_time_tensor, target_time_tensor):

		with tf.variable_scope('encoder'):	

			self.model_['encoder_w1'] = tf.Variable(tf.normal.truncated_normal([self.input_shape, 40], mean=0.0, stddev=1.0))
			self.model_['encoder_b1'] = tf.Variable(tf.zeros([1 ,40]))
			self.model_['encoder_a1'] = tf.nn.relu(tf.matmul(input_tensor, self.model_['encoder_w1']) + self.model_['encoder_b1'])

		with tf.variable_scope('decoder'):

			self.model_['latent_agg'] = tf.concat([input_tensor, self.model_['encoder_a1'], 
				source_time_tensor, target_time_tensor], axis=1)

			self.model_['decoder_w1'] = tf.Variable(tf.normal.truncated_normal([self.model_['latent_agg'].shape[1], self.input_shape], mean=0.0, stddev=1.0))
			self.model_['decoder_b1'] = tf.Variable(tf.zeros([1, self.model_['latent_agg'].shape[1]]))
			self.model_['output'] = tf.nn.relu(tf.matmul(self.model_['latent_agg'], self.model_['decoder_w1']) + self.model_['decoder_b1'])

		with tf.variable_scope('discriminator'):

			self.disc_['h1'] = tf.compat.v1.layers.dense()
	



