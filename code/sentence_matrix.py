import numpy as np
import cPickle as cp

''' Function for loading file data '''
def load_data(file_name):
	data = cp.load(open(file_name , 'r'))
	return data

''' 
We need to convert the list of words in sentence to it's vector equivalent form.
So in this function we will replace the words by their corresponding word vectors.
I am using GLOVE for this purpose
'''	
def prepare_sentence_matrix(file_name_in , file_name_out):
	# load the word vector model
	word_vec = load_data("../../word_vector/word2vec.cp")
	
	# load the raw_train_X dataset
	X = load_data(file_name_in)
	
	train_X = []
	for each in X:
		print(each)
		# Create an array for size 30 * 300 filled with all zeros.
		word_embed = np.zeros((30 , 300))
		
		for i in range(0 , len(each)):
			word = each[i]
			print(word)
			# Check if the word is present in the model or not. 
			# I have fixed the length of each sentence to 30. S0 check if index does not exceed that.
			if word in word_vec and i < 30:
				word_embed[i] = np.array(word_vec[word])
			else:
				print(each)
			
		train_X.append(word_embed)
	cp.dump(train_X , open(file_name_out , 'wb'))	


if __name__ == "__main__":
	prepare_sentence_matrix("raw_train_X.cp" , "train_X_final.cp")
