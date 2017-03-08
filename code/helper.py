import numpy as np
import cPickle
from spacy.en import English

''' Function to load the predefined word vectors ''' 
def loadGloveModel(gloveFile):
	embeddings_index = {}
	f = open(gloveFile)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	with open('word2vec.cp', 'wb') as word_vector_file:
		cPickle.dump(embeddings_index, word_vector_file, protocol=cPickle.HIGHEST_PROTOCOL)
	return embeddings_index


if __name__ == "__main__":	
	word_vec = loadGloveModel("glove.42B.300d.txt")
