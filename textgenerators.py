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

	if DEBUG:

		print('RUNNING IN DEBUGGING MODE!')


		for look_back in [4,8,20]:

			print('-'*60)
			print('Preparing data for look_back of %d'%look_back)
			data_train, data_val, data_test,emb_matrix,w2t,t2w=process_data(look_back=look_back, debug=DEBUG)


			#LSTM Euclid loss

			for nb_layers in [1,2,4]:

				tf.reset_default_graph()
				model=deep_LSTM_euclid.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_euclid_layers_'+str(nb_layers)+'_look_back_'+str(look_back))

			#LSTM Cross Entropy loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_cross_entropy.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_cross_entropy_layers_'+str(nb_layers)+' look_back_'+str(look_back))

			#LSTM NCE loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_nce.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_nce_layers_'+str(nb_layers)+' look_back_'+str(look_back))



	else:
		for look_back in [4,8,20]:

			print('-'*60)
			print('Preparing data for look_back of %d'%look_back)
			data_train, data_val, data_test,emb_matrix,w2t,t2w=process_data(look_back=look_back, debug=DEBUG)


			#LSTM Euclid loss

			for nb_layers in [1,2,4]:

				tf.reset_default_graph()
				model=deep_LSTM_euclid.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_euclid_layers_'+str(nb_layers)+'_look_back_'+str(look_back))

			#LSTM Cross Entropy loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_cross_entropy.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_cross_entropy_layers_'+str(nb_layers)+' look_back_'+str(look_back))

			#LSTM NCE loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_nce.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_nce_layers_'+str(nb_layers)+' look_back_'+str(look_back))

 

	

if __name__=='__main__':
	main()
