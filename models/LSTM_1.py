import numpy as np 
import tensorflow as tf 
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import gensim.models.word2vec as w2v
import os

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
    	pass

def create_corpus(source):
	"""Creat corpus from the source

	Args:
		source (list(string)): List of files 
	
	Returns:
		string: A large string containing all the text from the files	
	
	"""
	return " ".join([file.read() for file in source])

def create_emb_model(corpus):
	"""Uses word2vec to create a word embedding for the corpus and 
		gives back dictionaries for token to word and word 2 token 

	Args:
		corpus (strin): text with the data for the embedding

	Returns:
		numpy.ndarray,dict,dict: an embedding matrix, a word 2 token,
		 and token 2 word dictionaries.
	"""
	w2t={}
	t2w={}
	emb_matrix=[]
	raw_sentences = sent_tokenize(corpus)
	sentences=[]
	for sentence in raw_sentences:
	    sentences+=[word_tokenize(sentence)]

	emb_model=w2v.Word2Vec(sentences)

	for i,word in enumerate(emb_model.wv.vocab.keys()):
		w2t[word]=i
		t2w[i]=word
		emb_matrix+=[emb_model[[word]][0]]

	emb_matrix=np.array(emb_matrix)
	
	return emb_matrix,w2t,t2w


def all_w2t(w2t,corpus):
	return [w2t[word] for word in word_tokenize(corpus) if word in w2t]
	
def create_data(batch_size,look_back,tokens_data):

	X_=[tokens_data[i:look_back+i] for i in range(len(tokens_data)-look_back)]
	data = np.array(list(zip(X_[:-1],X_[1:])))
	np.random.seed(13)
	np.random.shuffle(data)
	l=len(data)
	data_train,data_val,data_test=data[:l*7//10],data[l*7//10:l*9//10],data[l*9//10:]
	
	X_train_,y_train_=zip(*data_train)
	X_train=[X_train_[i*batch_size:(i+1)*batch_size] for i in range(len(data_train)//batch_size)]
	y_train=[y_train_[i*batch_size:(i+1)*batch_size] for i in range(len(data_train)//batch_size)]


	X_val_,y_val_=zip(*data_val)
	X_val=[X_val_[i*batch_size:(i+1)*batch_size] for i in range(len(data_val)//batch_size)]
	y_val=[y_val_[i*batch_size:(i+1)*batch_size] for i in range(len(data_val)//batch_size)]


	X_test_,y_test_=zip(*data_test)
	X_test=[X_test_[i*batch_size:(i+1)*batch_size] for i in range(len(data_test)//batch_size)]
	y_test=[y_test_[i*batch_size:(i+1)*batch_size] for i in range(len(data_test)//batch_size)]

	

	return zip(X_train,y_train), zip(X_val,y_val), zip(X_test,y_test)




class LSTMmodel:
	"""Build the graph for the model"""
	def __init__(self,look_back=4,emb_matrix=None,batch_size=1000,lr=0.01,nb_layers=1,vocab_size=None,embed_size=None):
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

SKIP_STEP=5
def train_model(model, tokens_data,batch_size=1000, look_back=4,nb_train_steps=1,folder_to_save='./temp'):
	saver = tf.train.Saver()

	initial_step=0
	make_dir(folder_to_save)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(folder_to_save)
		
		if ckpt:
			saver.restore(sess,ckpt.model_checkpoint_path)

		writer = tf.summary.FileWriter(folder_to_save+'/improved_graph', sess.graph)
		initial_step = model.global_step.eval()
		
		print("The initial step is %d"%initial_step)

		train_data,val_data,test_data=create_data(batch_size,look_back,tokens_data)
		train_data=list(enumerate(train_data))
		val_data=list(val_data)
		#test_data=list(enumerate(test_data))


		for i in range(initial_step,initial_step+nb_train_steps):
			for j,data_Xy in train_data:
				global_step=i*len(train_data)+j
				print("Epoch %d out of %d, step %d"%(i+1,nb_train_steps,global_step),end='\r')
				X_batch,y_batch = data_Xy
				feed_dict={model.input_words:X_batch,model.output_words:y_batch}
				loss_batch,_,summary = sess.run([model.loss,model.optimizer,model.summary_train_op],feed_dict=feed_dict)
				writer.add_summary(summary, global_step=global_step)

				X_batch,y_batch=val_data[np.random.randint(0,len(val_data))]
				feed_dict={model.input_words:X_batch,model.output_words:y_batch}
				loss_batch,summary = sess.run([model.loss,model.summary_val_op],feed_dict=feed_dict)
				writer.add_summary(summary, global_step=global_step)


			if (i+1)%SKIP_STEP==0:
				saver.save(sess, folder_to_save, global_step)
		print("\n")

def preparations(test=True):
	if test:
		books=[open('./data/dorian.txt','r')]
	else:
		books=[open('./data/dorian.txt','r'),open('./data/earnest.txt','r'),
	       open('./data/essays.txt','r'),open('./data/ghost.txt','r'),
	       open('./data/happy_prince.txt','r'),open('./data/house_pomegranates.txt','r'),
	       open('./data/ideal_husband.txt','r'),open('./data/intentions.txt','r'),
	       open('./data/lady_windermere.txt','r'),open('./data/profundis.txt','r'),
	       open('./data/salome.txt','r'),open('./data/soul_of_man.txt','r'),
	       open('./data/woman_of_no_importance.txt','r')]
	corpus=create_corpus(books)
	emb_matrix,w2t,t2w = create_emb_model(corpus)
	tokens_data=all_w2t(w2t,corpus)

	return emb_matrix,w2t,t2w,tokens_data

def _compute(tokens_data,look_back=4,emb_matrix=None,batch_size=1000,lr=0.01,nb_layers=1,nb_train_steps=2,folder_to_save='./temp'):
	tf.reset_default_graph()
	model=LSTMmodel(look_back=look_back,emb_matrix=emb_matrix,batch_size=batch_size,lr=lr,nb_layers=nb_layers)
	model.build_graph()
	train_model(model, tokens_data,batch_size=batch_size, look_back=look_back,nb_train_steps=nb_train_steps,folder_to_save=folder_to_save)


def compute():
	#computes several models

	emb_matrix,w2t,t2w,tokens_data=preparations(test=False)
	
	print('model 1 is being computed')
	_compute(tokens_data,look_back=4,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=1,nb_train_steps=5,folder_to_save='./model1')
	print('model 2 is being computed')
	_compute(tokens_data,look_back=4,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=2,nb_train_steps=5,folder_to_save='./model2')
	print('model 3 is being computed')
	_compute(tokens_data,look_back=4,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=3,nb_train_steps=5,folder_to_save='./model3')
	print('model 4 is being computed')
	_compute(tokens_data,look_back=4,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=4,nb_train_steps=5,folder_to_save='./model4')

	print('model 5 is being computed')
	_compute(tokens_data,look_back=8,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=1,nb_train_steps=5,folder_to_save='./model5')
	print('model 6 is being computed')
	_compute(tokens_data,look_back=8,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=2,nb_train_steps=5,folder_to_save='./model6')
	print('model 7 is being computed')
	_compute(tokens_data,look_back=8,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=3,nb_train_steps=5,folder_to_save='./model7')
	print('model 8 is being computed')
	_compute(tokens_data,look_back=8,emb_matrix=emb_matrix,batch_size=1000,lr=0.01,nb_layers=4,nb_train_steps=5,folder_to_save='./model8')


if __name__ == "__main__":
	
	compute()






    