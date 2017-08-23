"""Creates a handful of models for text generations and shows some sample results
Author: Felipe Perez
Just for fun
scy1505.github.io/
"""

import os
import utils
import tensorflow as tf 
import numpy as np 
import time
from process_data import process_data
from models import deep_LSTM_euclid,deep_LSTM_cross_entropy,deep_LSTM_nce
from params import *
from nltk.tokenize import word_tokenize


def main():


	date = time.localtime()

	log_name='./data/logs/log_'
	log_name+=str(date.tm_mon)+'_'
	log_name+=str(date.tm_mday)+'_'
	log_name+=str(date.tm_hour)+'_'
	log_name+=str(date.tm_min)+'_'
	log_name+=str(date.tm_sec)+'.txt'

	log_file=open(log_name,'w')

	current_message=open('./data/starting_message.txt','r').read()
	log_file.write(current_message)
	print(current_message)

	beginnings_file=open('./data/starting_sentences.txt','r')

	beginnings=[word_tokenize(beginning) for beginning in beginnings_file.readlines()]

	
	if DEBUG:

		print('RUNNING IN DEBUGGING MODE!')


		for look_back in [4,8,20]:


			current_message='-'*60+'\n'
			current_message+='Preparing data for look_back of %d'%look_back+'\n'
			log_file.write(current_message)
			print(current_message)

			start_time=time.time()
			
			data_train, data_val, data_test,emb_matrix,w2t,t2w,emb_model=process_data(log_file=log_file,look_back=look_back, debug=DEBUG)

		
			current_message="Data took %.2f seconds to prepare." % (time.time() - start_time)+'\n'
			current_message+='\n'+'-'*60+'\n'
			log_file.write(current_message)
			print(current_message)
			
			#LSTM Euclid loss

			for nb_layers in [1,2,4]:

				start_time=time.time()

				tf.reset_default_graph()
				model=deep_LSTM_euclid.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_euclid_layers_'+str(nb_layers)+'_look_back_'+str(look_back))

				current_message="Model Euclid with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				for beginning in beginnings:
					model.create_story(emb_model=emb_model,w2t=w2t,t2w=t2w,beginning=beginning)


			#LSTM Cross Entropy loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_cross_entropy.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_cross_entropy_layers_'+str(nb_layers)+' look_back_'+str(look_back))

				current_message="Model Entropy with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				for beginning in beginnings:
					model.create_story(w2t=w2t,t2w=t2w,beginning=beginning)

			#LSTM NCE loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_nce.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=1,
					folder_to_save='results/LSTM_nce_layers_'+str(nb_layers)+' look_back_'+str(look_back))

				current_message="Model NCE with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				for beginning in beginnings:
					model.create_story(w2t=w2t,t2w=t2w,beginning=beginning)


	else:

		for look_back in [4,8,20]:

			current_message='-'*60+'\n'
			current_message+='Preparing data for look_back of %d'%look_back
			log_file.write(current_message)
			print(current_message)



			start_time=time.time()
			
			data_train, data_val, data_test,emb_matrix,w2t,t2w=process_data(look_back=look_back, debug=DEBUG)



			current_message="Data took %.2f seconds to prepare." % (time.time() - start_time)+'\n'
			current_message+='\n'+'-'*60+'\n'
			log_file.write(current_message)
			print(current_message)

			#LSTM Euclid loss

			for nb_layers in [1,2,4]:

				start_time=time.time()

				tf.reset_default_graph()
				model=deep_LSTM_euclid.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,
					nb_layers=nb_layers,log_file=log_file)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_euclid_layers_'+str(nb_layers)+'_look_back_'+str(look_back))
				
				current_message="Model Euclid with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				
				for beginning in beginnings:
					model.create_story(emb_model=emb_model,w2t=w2t,t2w=t2w,beginning=beginning)


			#LSTM Cross Entropy loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_cross_entropy.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_cross_entropy_layers_'+str(nb_layers)+' look_back_'+str(look_back))

				current_message="Model Entropy with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				for beginning in beginnings:
					model.create_story(w2t=w2t,t2w=t2w,beginning=beginning)



			#LSTM NCE loss

			for nb_layers in [1,2,4]:
				
				tf.reset_default_graph()
				model=deep_LSTM_nce.LSTMmodel(emb_matrix=emb_matrix,look_back=look_back,nb_layers=nb_layers)
				model.build_graph()
				model.train(data_train,data_val,nb_train_steps=5,
					folder_to_save='results/LSTM_nce_layers_'+str(nb_layers)+' look_back_'+str(look_back))

				current_message="Model NCE with %d layers took %.2f for building and training." % (nb_layers,time.time() - start_time)
				current_message+='\n'+'-'*40+'\n'
				log_file.write(current_message)
				print(current_message)

				for beginning in beginnings:
					model.create_story(w2t=w2t,t2w=t2w,beginning=beginning)


if __name__=='__main__':
	main()
