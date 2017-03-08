import numpy as np
import cPickle as cp
import csv
from spacy.en import English
from keras.models import load_model
from format_result import format_result

''' Function to read the tsv file '''
def read_tsv(file_name):
	# Create an empty list to store the result
	test_X = []	
	
	with open(file_name , "r") as data_file:
		data = csv.reader(data_file , delimiter = '\t')
		for row in data:
			test_X.append(row[0])
	return test_X
	
''' Function to tokenize a sentence into list of words '''
def word_tokenizer(model , doc):
	# Create an empty list to store the store the tokenized words
	word_list = []
	
	for each in doc:
		# decode the string to utf-8 format and send it to spacy model 
		text = model(each.decode("utf-8"))
		
		words = []
		for token in text:
			words.append(token.lemma_)
		word_list.append(words)
	return word_list
	
''' Function to tokenize raw text into list of sentences '''
def sentence_tokenizer(text):
	sent_list = sent_tokenize(text)
	return sent_list
	
''' Function to load data from cPickle format '''
def load_data(file_name):
	data = cp.load(open(file_name , 'r'))
	return data

''' Function to prepare sentence matrix from tokenized sentences '''
def prepare_sentence_matrix(word_list):
	# load the word vector model
	word_vec = load_data("../../word_vector/word2vec.cp")
	
	test_X = []
	for each in word_list:
		# numpy array of dim 30*300 filled with all zeros
		word_embed = np.zeros((30 , 300))
		
		for i in range(0 , len(each)):
			word = each[i]
			if word in word_vec and i < 30:
				word_embed[i] = np.array(word_vec[word])
			else:
				print(word)
		test_X.append(word_embed)
	return test_X

''' Function to create chunks of window 7 '''
def create_chunks(temp_X):
	test_X = []
	for each in temp_X:
		X = []
		chunk = np.zeros((36,300))
		for j in range(0 , len(each)):
			chunk[j+3] = each[j]
		
		# logic to create artificial bias in the chunks of window 7.
		# Central element is of utmost importance to us than it's neighbours.
		# By multiplying the vectors by constants, we are creating artificial bias.	
		for j in range(0 , len(each)):
		
			temp = chunk[j:j+7]
			temp[1] = temp[1]*3
			temp[5] = temp[5]*3
			temp[2] = temp[2]*7
			temp[4] = temp[4]*7
			temp[3] = temp[3]*9
				
			X.append(temp)
		test_X.append(X)		
	return test_X

if __name__ == "__main__":	
	text = read_tsv("eval_data.txt")
	print(text)
	model = load_model('model/weight_2.h5')
	nlp = English()
	
	# 2 dimensional array containing list of words for all sentences
	word_list = word_tokenizer(nlp , text)
	
	# prepare sentence matrix for the tokenized sentences
	test_X = prepare_sentence_matrix(word_list)
	
	# create chunks of window 7
	X = create_chunks(test_X)
	
	# convert the list to numpy array
	X = np.array(X)
	
	# create an empty list to store model's predictions
	pred = []
	
	for each in X:
		pred.append(model.predict(each))
	
	# store the list in files in cpickle format
	cp.dump(pred , open("pred_2.cp" , 'w'))
	cp.dump(word_list , open("word_list_2.cp" , 'w'))
	
	
	format_result("word_list_2.cp" , "pred_2.cp" , "result_final.csv")
			
		
	
