import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np 
from utils import make_dir
from params import *

class LSTMmodel:
	"""Build the graph for the model"""

	def __init__(self,emb_matrix,look_back=4,batch_size=1024,lr=0.01,nb_layers=1):
		

		self.emb_matrix=emb_matrix

		self.look_back=look_back
		self.vocab_size=self.emb_matrix.shape[0]
		self.emb_size=self.emb_matrix.shape[1]
		self.batch_size=batch_size
		self.lr=lr
		self.nb_layers=nb_layers
		self.global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")


	def _create_placeholders(self):
		"""Creates placeholders for input and output"""

		with tf.name_scope("input_data"):
			self.input_words=tf.placeholder(shape=(None,self.look_back), dtype=tf.int32,name='input_tokens')
		with tf.name_scope("output_data"):	
			self.output_words=tf.placeholder(shape=(None,self.look_back),dtype=tf.int32,name='output_tokens')


	def _create_embedding(self,trainable=False):
		with tf.name_scope("embedding"):
			self.emb_matrix=tf.Variable(self.emb_matrix,trainable=trainable,dtype=tf.float32)


	def _create_recurrent_layers(self):
		_output = tf.nn.embedding_lookup(self.emb_matrix, self.input_words, name='embed_inputs')

		with tf.name_scope("recurrent_layers"):
			self.lstms=[tf.contrib.rnn.LSTMCell(self.emb_size) for i in range(self.nb_layers)]
			self.stacked_lstm=tf.contrib.rnn.MultiRNNCell(
				self.lstms)
			initial_state=state = self.stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
			_output,_=tf.nn.dynamic_rnn(self.stacked_lstm,_output,dtype=tf.float32)
			self.pred_output=_output

	def _create_de_embedding(self,trainable=True):
		
		with tf.name_scope("de_embedding"):

			self.nce_weights=tf.Variable(tf.random_normal([self.vocab_size,self.emb_size], stddev=1.0 / self.emb_size**0.5),trainable=trainable,dtype=tf.float32)
			self.nce_bias=tf.Variable(tf.zeros([self.vocab_size]),dtype=tf.float32)

			self.pred_output_computed=tf.matmul(self.pred_output[:,-1,:],tf.transpose(self.nce_weights))+self.nce_bias

	def _create_loss(self):

		self.output_vectors= tf.reshape(self.output_words[:,-1],shape=(-1,1))
		self.pred_output=self.pred_output[:,-1,:]

		self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,biases=self.nce_bias,
			labels=self.output_vectors,inputs=self.pred_output,
			num_sampled=int(0.1*self.vocab_size),num_classes=self.vocab_size,num_true=1))


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
		self._create_de_embedding()
		self._create_loss()
		self._create_optimizer()
		self._create_summaries()


	def train(self, train_data, val_data, nb_train_steps=1,folder_to_save='temp'):



		folder_to_save=make_dir(folder_to_save)

		self.folder_to_save=folder_to_save

		print('Training LSTM with NCE loss.')

		#This is not the right way, but meanwhile
		print('Model will be save at ./'+folder_to_save[folder_to_save.find('tion/')+5:])


		saver = tf.train.Saver()
		#initial_step=0

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			ckpt = tf.train.get_checkpoint_state(folder_to_save)

			if ckpt:
				saver.restore(sess,ckpt.model_checkpoint_path)

			writer = tf.summary.FileWriter(folder_to_save+'/improved_graph', sess.graph)
			

			train_data=list(enumerate(train_data))


			initial_step = self.global_step.eval()//len(train_data)

			print("The initial step is %d"%initial_step)


			for i in range(initial_step,initial_step+nb_train_steps):
				for j,data_Xy in train_data:

					global_step=i*len(train_data)+j
					print("Epoch %d out of %d, step %d"%(i+1,nb_train_steps+initial_step,global_step),end='\r')

					X_batch,y_batch = data_Xy
					
					feed_dict={self.input_words:X_batch,self.output_words:y_batch}
					loss_batch,_,summary = sess.run([self.loss,self.optimizer,self.summary_train_op],feed_dict=feed_dict)
					writer.add_summary(summary, global_step=global_step)

					X_batch,y_batch=val_data
					feed_dict={self.input_words:X_batch,self.output_words:y_batch}
					loss_batch,summary = sess.run([self.loss,self.summary_val_op],feed_dict=feed_dict)
					writer.add_summary(summary, global_step=global_step)


				if (i+1)%SKIP_STEP==0:
					saver.save(sess, folder_to_save+"/step", global_step)
			print("\n")


	def create_story(self,w2t,t2w,beginning):

		story=[w2t[word] if word in w2t else 0 for word in beginning ][:self.look_back]

		folder_to_save=self.folder_to_save
		saver = tf.train.Saver()

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			ckpt = tf.train.get_checkpoint_state(folder_to_save)

			if ckpt:
				saver.restore(sess,ckpt.model_checkpoint_path)

			for i in range(STORY_LENGTH):

				X_batch=[story[-self.look_back:]]
				feed_dict={self.input_words:X_batch}

				next_token=sess.run(self.pred_output_computed,feed_dict=feed_dict)
				next_word=np.argmax(next_token[0])

				#print(next_word.shape)

				story+=[next_word]
		
		stories_file=open(folder_to_save+'/story.txt','a')
		story = ' '.join([t2w[token] for token in story])
		stories_file.write(story+'\n \n')



