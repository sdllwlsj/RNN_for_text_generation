import tensorflow as tf
import numpy as np 

class LSTMmodel:
	"""Build the graph for the model"""
	def __init__(self,look_back=4,emb_matrix=None,batch_size=1024,lr=0.01,nb_layers=1,vocab_size=None,embed_size=None):
		self.look_back=look_back
		if type(emb_matrix) is np.ndarray:
			self.emb_matrix=emb_matrix
		else:
			self.emb_matrix=tf.truncated_normal([self.vocab_size, self.embed_size],
                                                            stddev=1.0 / (self.embed_size ** 0.5))
		self.vocab_size=self.emb_matrix.shape[0]
		self.emb_size=self.emb_matrix.shape[1]
		self.batch_size=batch_size
		self.lr=lr
		self.nb_layers=nb_layers
		self.global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")


	def _create_placeholders(self):
		"""Creates placeholders for input and output"""
		with tf.name_scope("input_data"):
			self.input_words=tf.placeholder(shape=(self.batch_size,self.look_back),dtype=tf.int32,name='input_tokens')
		with tf.name_scope("output_data"):	
			self.output_words=tf.placeholder(shape=(self.batch_size,self.look_back),dtype=tf.int32,name='output_tokens')

	def _create_embedding(self,trainable=True):
		with tf.name_scope("embedding"):
			self.emb_matrix=tf.Variable(self.emb_matrix,dtype=tf.float32)

	def _create_recurrent_layers(self):
		_output = tf.nn.embedding_lookup(self.emb_matrix, self.input_words, name='embed_inputs')

		with tf.name_scope("recurrent_layers"):
			self.lstms=[tf.contrib.rnn.LSTMCell(self.emb_size) for i in range(self.nb_layers)]
			self.stacked_lstm=tf.contrib.rnn.MultiRNNCell(
				self.lstms)
			initial_state=state = self.stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
			_output,_=tf.nn.dynamic_rnn(self.stacked_lstm,_output,dtype=tf.float32)
			self.pred_output=_output

	def _create_loss(self):
		self.output_vectors= tf.nn.embedding_lookup(self.emb_matrix, self.output_words, name='embed_outputs')
		self.loss = tf.reduce_mean(tf.square(self.pred_output-self.output_vectors))


	def _create_optimizer(self):
		self.optimizer=tf.train.AdamOptimizer().minimize(self.loss,global_step=self.global_step)

	def _create_summaries(self):
		with tf.name_scope("summaries"):
			self.train_loss_summary=tf.summary.scalar("train_loss", self.loss)
			self.train_loss_histogram=tf.summary.histogram("histogram_train_loss", self.loss)
			self.summary_train_op = tf.summary.merge((self.train_loss_summary,self.train_loss_histogram))
			self.val_loss_summary=tf.summary.scalar("val_loss",self.loss)
			self.val_loss_histogram=tf.summary.histogram("histogram_val_loss", self.loss)
			self.summary_val_op = tf.summary.merge((self.val_loss_summary,self.val_loss_histogram))


	def build_graph(self):
		"""Build graph for the model"""
		self._create_placeholders()
		self._create_embedding()
		self._create_recurrent_layers()
		self._create_loss()
		self._create_optimizer()
		self._create_summaries()
