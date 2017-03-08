from __future__ import unicode_literals
import csv
import cPickle as cp
import numpy as np
from spacy.en import English

# Global variable
NUM_WORDS = 30

# There are total 9819 sentences and 9819 phrases. No missing value reported
''' Function to read tsv '''
def read_tsv(file_name):
	train_X = []
	train_Y = []
	with open(file_name , "r") as data_file:
		data = csv.reader(data_file , delimiter = '\t')
		for row in data:
			train_X.append(row[0])
			train_Y.append(row[1])
	
	
	print(len(train_X) , len(train_Y))
	cp.dump(train_X , open("list_sent.cp" , 'wb'))
	cp.dump(train_Y , open("list_phrase.cp" , 'wb'))

''' Function for tokenizing the words, given a sentence '''
def word_tokenizer(model , doc):
	text = model(doc.decode("utf-8"))
	words = []
	for token in text:
		words.append(token.lemma_)
		print(token.lemma_)
	return words

''' Function to prepare data for training '''
def prepare_training_data(sent_list_file , phrase_list_file):
	
	X_temp = []	# a list to hold temporary X data
	Y_temp = []	# a list to hold temporary Y data
	NF_temp = [] # a list to hold 'Not found' data
	index_len_list = []	# a list to hold the number of words in a sentence
	index_len_list_NF = []	# a list to hold the number of words in NF
	
	# Load spacy model for word tokenization
	model = English()
	
	# Read the sentence list and phrase list files in two variables
	X = cp.load(open(sent_list_file, 'r'))
	Y = cp.load(open(phrase_list_file, 'r'))
	
	
	# iterate over the list in Y and check if 'Not found' is available as a phrase
	for i in range(0 , len(Y)):
		if Y[i] != 'Not Found':
			# There exist a list of words that are phrases
			# Tokenize the sentence both in X and Y
			tokenized_word_list = word_tokenizer(model , X[i])
			index_len_list.append(len(tokenized_word_list))
			X_temp.append(tokenized_word_list)
			Y_temp.append(word_tokenizer(model , Y[i]))
		else:
			tokenized_word_list = word_tokenizer(model , X[i])
			index_len_list_NF.append(len(tokenized_word_list))
			NF_temp.append(tokenized_word_list)
			
			
	train_Y = []
	for i in range(0,len(X_temp)):
		
		# let us fix the length of words contained in a sentence to be 30
		label = np.zeros()
		
		word_list = X_temp[i]
		phrase_list = Y_temp[i]
		
		for j in range(0,len(word_list)):
			# if the word exist in the phrase list and is with valid index limit, change the label from 0 to 1.
			if word_list[j] in phrase_list and j<NUM_WORDS:
				label[j] = 1
				
		train_Y.append(label)
		
	cp.dump(X_temp , open("raw_train_X.cp" , 'wb'))
	cp.dump(train_Y , open("final_train_Y.cp" , 'wb'))
	cp.dump(NF_temp , open("not_found_sent_list.cp" , 'wb'))
	cp.dump(index_len_list , open("index_len_list.cp" , 'wb'))
	cp.dump(index_len_list_NF , open("index_len_list_NF.cp" , 'wb'))
		
if __name__ == "__main__":		
	read_tsv("training_data.tsv")
	
	# call the function to tokenize the sentences and prepare data
	prepare_training_data("list_sent.cp" , "list_phrase.cp")
