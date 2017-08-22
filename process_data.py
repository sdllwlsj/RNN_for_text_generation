import os
import numpy as np 
from utils import create_emb_model
from nltk.tokenize import word_tokenize
from params import *


def create_corpus(source):
	"""Creat corpus from the source

	Args:
		source (list(string)): List of files 
	
	Returns:
		string: A large string containing all the text from the files	
	
	"""
	return " ".join([file.read() for file in source])



def process_data(look_back=4,batch_size=1024,split=[0.7,0.2,0.1],debug=False):
	"""process the data from the /data/files folder

	Args:
		look_back (int): The number of previous words we are going to use.
		batch_size (int): batch_size.
		split list(float): list of three floats correspoding to the percentage of
							data use for training, validation, and splitting. 
	"""
	if (len(split)!=3 or abs(sum(split)-1)>0.0001):
		raise ValueError('Need three values adding to one.')

	path=os.getcwd()
	file_names=os.listdir(path+'/data/files')

	print('The following files will be used:')
	
	source=[]
	for file in file_names:
		print(file)
		source.append(open('./data/files/'+file,'r'))
		if debug:
			break
	print('\n'+'-'*40)
	corpus=create_corpus(source)

	emb_matrix,w2t,t2w = create_emb_model(corpus,RANDOM_EMB)

	corpus_tokens=[w2t[word] for word in word_tokenize(corpus) if word in w2t]
	
	data=np.array([corpus_tokens[:-1],corpus_tokens[1:]]).T
	
	data=np.array([data[i:i+look_back] for i in range(len(data)-look_back)])

	np.random.seed(13)
	np.random.shuffle(data)

	l=len(data)

	data_train,data_val,data_test=data[:int(l*split[0])],data[int(l*split[0]):int(l*(split[0]+split[1]))],data[int(l*(split[0]+split[1])):]
	
	data_train=np.array([data_train[i*batch_size:(i+1)*batch_size] for i in range(len(data_train)//batch_size)])
	

	data_train=np.swapaxes(np.swapaxes(data_train,1,3),2,3)
	
	data_val=np.swapaxes(np.swapaxes(data_val,0,2),1,2)

	data_test=np.swapaxes(np.swapaxes(data_test,0,2),1,2)

	#Uncomment the following if you want to train the val and test by batches.
	#data_val=np.array([data_val[i*batch_size:(i+1)*batch_size] for i in range(len(data_val)//batch_size)])
	#data_test=np.array([data_test[i*batch_size:(i+1)*batch_size] for i in range(len(data_test)//batch_size)])

	print('There are %d words in the vocabulary.'%len(w2t))

	return data_train,data_val,data_test,emb_matrix,w2t,t2w


	