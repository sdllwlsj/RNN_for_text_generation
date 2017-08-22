"""Creates a handful of models for text generations and shows some sample results
Author: Felipe Perez
Just for fun
scy1505.github.io/
"""

import os
import utils
import tensorflow as tf 
import numpy as np 
from process_data import process_data
from models import deep_LSTM_euclid,deep_LSTM_cross_entropy,deep_LSTM_nce
from params import DEBUG,SEED,SKIP_STEP, RANDOM_EMB


def main():
	print(open('./data/starting_message.txt','r').read())

	print('Preparing models for look_back of 4')

	data_train, data_val, data_test,emb_matrix,w2t,t2w=process_data(debug=DEBUG)


	if not DEBUG:
		for nb_layers in range(1,5):


			tf.reset_default_graph()

			model=deep_LSTM_euclid.LSTMmodel(emb_matrix=emb_matrix,nb_layers=nb_layers)
			model.build_graph()

			model.train(data_train,data_val,nb_train_steps=5,folder_to_save='temp'+str(nb_layers))


	else:
		for nb_layers in range(1,5):


			tf.reset_default_graph()

			model=deep_LSTM_nce.LSTMmodel(emb_matrix=emb_matrix,nb_layers=nb_layers)
			#model=deep_LSTM_cross_entropy.LSTMmodel(emb_matrix=emb_matrix,nb_layers=nb_layers)
			model.build_graph()
			model.train(data_train,data_val,nb_train_steps=5,folder_to_save='temp'+str(nb_layers))

			break

	

if __name__=='__main__':
	main()
