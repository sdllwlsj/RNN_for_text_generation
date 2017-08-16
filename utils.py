import numpy as np 
from nltk.tokenize import word_tokenize,sent_tokenize
import gensim.models.word2vec as w2v


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

