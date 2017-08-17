import numpy as np 
import os 
from nltk.tokenize import word_tokenize,sent_tokenize
import gensim.models.word2vec as w2v


def create_emb_model(corpus,random=False,emb_size=100):
	"""Uses word2vec to create a word embedding for the corpus and 
		gives back dictionaries for token to word and word 2 token 

	Args:
		corpus (string): text with the data for the embedding.
		ramdom (bool): creates a random embedding.

	Returns:
		numpy.ndarray,dict,dict: an embedding matrix, a word 2 token,
		 and token 2 word dictionaries.
	"""


	if random:
		return _create_emb_model_random(corpus,emb_size)
	else:
		return _create_emb_model(corpus,emb_size)


def _create_emb_model(corpus,emb_size):
	
	w2t={}
	t2w={}
	emb_matrix=[]
	raw_sentences = sent_tokenize(corpus)
	sentences=[]
	for sentence in raw_sentences:
	    sentences+=[word_tokenize(sentence)]

	emb_model=w2v.Word2Vec(sentences,size=100)

	for i,word in enumerate(emb_model.wv.vocab.keys()):
		w2t[word]=i
		t2w[i]=word
		emb_matrix+=[emb_model[[word]][0]]

	emb_matrix=np.array(emb_matrix)
	
	return emb_matrix,w2t,t2w


def _create_emb_model_random(corpus,emb_size):

	import tensorflow as tf 

	t2w=dict(enumerate(set(word_tokenize(corpus))))
	w2t={v: k for k, v in t2w.items()}

	emb_matrix=np.random.random(size=[len(w2t), emb_size])

	return emb_matrix,w2t,t2w



def make_dir(name):
    """ Create a directory if there isn't one already. """
    path=os.getcwd()+'/'+name
    try:
        os.mkdir(path)
    except OSError:
    	pass
    return path