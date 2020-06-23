import numpy as np
import tensorflow as tf

class CVAE():

	def __init__(self, input_shape, batch_size=10, hidden_layers=1, dims=None):

		if dims == None: dims = [input_shape/2]
		assert(len(dims) == hidden_layers+1)

		self.input_shape = input_shape
		self.hidden_layers = hidden_layers
		self.dims = dims
		self.model_ = dict()
		self.loss_ = None


	def init_model(self, input_tensor, cond_tensor, output_tensor, train_tensor):

		self.model_['cat_tensor'] = tf.concat([input_tensor, cond_tensor], axis=1)

		with tf.variable_scope('encoder'):

			self.model_['enc'] = tf.compat.v1.layers.dense(self.model_['cat_tensor'], self.dims[0], activation=tf.nn.relu)

			for i in range(1,self.hidden_layers):

				self.model_['enc'] = tf.compat.v1.layers.dense(self.model_['enc'], self.dims[i],  activation=tf.nn.relu)

			self.model_['mu'] = tf.compat.v1.layers.dense(self.model_['enc'], self.dims[-1])
			self.model_['log_sigma'] = tf.compat.v1.layers.dense(self.model_['enc'], self.dims[-1])


		#eps = tf.random.normal(shape=(batch_size, self.dims[-1]))
		eps = tf.random.normal(shape=tf.shape(self.model_['mu']))
		Z = self.model_['mu'] + tf.math.exp(self.model_['log_sigma']/2)*eps
		#Z = tf.cond(tf.math.equal(train_tensor,1), lambda: self.model_['mu'] + tf.math.exp(self.model_['log_sigma']/2)*eps, 
		#			lambda: eps)
		cat_Z = tf.concat([Z, cond_tensor], axis=1)

		with tf.variable_scope('decoder'):

			self.model_['dec'] = tf.compat.v1.layers.dense(cat_Z, self.dims[-2],  activation=tf.nn.relu)

			for i in range(self.hidden_layers-3,-1,-1):

				self.model_['dec'] = tf.compat.v1.layers.dense(self.model_['dec'], self.dims[i],  activation=tf.nn.relu)

			self.model_['op_tensor'] = tf.compat.v1.layers.dense(self.model_['dec'], self.input_shape,  activation=tf.nn.sigmoid)

		with tf.variable_scope('loss'):

			self.model_['recon_loss'] = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model_['op_tensor'], 
														labels=output_tensor), axis=1)
			self.model_['kl_loss'] = 0.5 * tf.reduce_sum(tf.math.exp(self.model_['log_sigma']) + self.model_['mu']**2 - 
						1.0 - self.model_['log_sigma'], axis=1)

			self.model_['net_loss'] = tf.reduce_mean(self.model_['recon_loss'] + self.model_['kl_loss'])
			#self.model_['net_loss'] = tf.reduce_sum(self.model_['recon_loss'] + self.model_['kl_loss'])

		self.loss_ = self.model_['net_loss']

	#def generate_sample(self, noise_tensor, cond_tensor)


		

