import cPickle as cp
import numpy as np
import random
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint , EarlyStopping , History 
from create_network import create_model
from keras.optimizers import SGD , Adam

''' Function to convert the training lists to respective numpy arrays '''
def format_training_data(file_X_in , file_X_out , file_Y_in , file_Y_out):
	# load already prepared data for training 
	train_X = cp.load(open(file_X , 'r'))
	train_Y = cp.load(open(file_Y , 'r'))
	
	# Convert the lists to numpy array for training
	train_X = np.array(train_X)
	train_Y = np.array(train_Y)
	
	# Print the sizes of the array and check if they are same
	np.save(file_X_out , train_X)
	np.save(file_Y_out , train_Y)


''' 
Function to divide whole data in the chunks of length 7.
I will multiply the contents of window by some constant to create artificial bias.
'''
def create_chunks(train_X , train_Y , index_len_file):
	X = []
	Y = []
	
	# load the index length file 
	len_list = cp.load(open(index_len_file , 'r'))
	
	for i in range(0 , len(train_X)):
		temp_X = train_X[i]
		temp_Y = train_Y[i]
		
		# a numpy array of dimension 36 * 300.
		# Goal is to divide the 30 words in the sentence in the chunks of window 7.
		# So a total of 36 words with a window of 7 will give exactly 30 chunks.
		# Each chunk contains the central word and 3 of it's neighbouring words from either side.
		# Justification for window of 7 is taken from G. Mesnil's and Yoshua Bengio's paper. 
		chunk = np.zeros((36,300))
		for j in range(0 , len(temp_X)):
			chunk[j+3] = temp_X[j]
		
		for j in range(0 , len_list[i]):
			if len_list[i] < 30:
				temp = chunk[j:j+7]
				'''
				Confidential Code
				'''
				
				# final class labelling for the chunks of window 7
				if temp_Y[j] == 0:
					Y.append([1,0])
				else:
					Y.append([0,1])
			
				X.append(temp)
	return X , Y

''' Function to randomly shuffle train_X and train_Y together '''
def process_train_data(train_X , train_Y):
	temp = list(zip(train_X , train_Y))
	random.shuffle(temp)
	train_X , train_Y = zip(*temp)
	train_X = np.array(train_X)
	train_Y = np.array(train_Y)
	return train_X , train_Y

''' Function to train the model '''
def train_model(train_X , train_Y):
	
	# We will store only the best possible model based on the val_loss. 
	# In turn it helps us avoiding overfitting.
	checkpoint = ModelCheckpoint("model/weight_1.h5" ,  monitor='val_loss' , verbose = 1 , save_best_only=True)
	
	# EarlyStopping is again a measure to avoid overfitting.
	es = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
	
	# Create the model which is define in create_network. 
	# Input dimention is 7*300 i.e. each chunk is sent to the model for training. 
	model = create_model((7 , 300))
	
	print model.summary()
	
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Compile the model with optimizer 'Adam' and loss 'binary_crossentropy'
	model.compile(loss='binary_crossentropy', optimizer='Adam')
	
	# save the model architecture
	json_string = model.to_json()
	fp = open("model/model_1_arch" , "w")
	fp.write(json_string)
	
	# train the model for 100 epochs with earlystopping and batch size 32
	model.fit(train_X , train_Y , batch_size = 32 , nb_epoch = 100 , verbose = 1 , callbacks = [checkpoint , es] , validation_split = .2 , shuffle = True , )
	

if __name__ == "__main__":
	#~ format_training_data("train_X_final.cp" , 'train_X.npy' , "train_Y_final.cp" , 'train_Y.npy' )
	
	# load the numpy arrays corresponding to train_X and train_Y
	train_X = np.load('train_X.npy')
	train_Y = np.load('train_Y.npy')
	
	# create the chunks of window 7
	X , Y = create_chunks(train_X , train_Y , "index_len_list.cp")
	
	# convert the list X and Y to numpy arrays
	X = np.array(X)
	Y = np.array(Y)
	
	# save the numpy arrays for future use
	np.save("chunk_X.npy" , X)
	np.save("chunk_y.npy" , Y)
	
	# finally attained train_X = (60647, 7, 300) , train_Y =  (60647, 2)
	print X.shape , Y.shape
	
	# Shuffle train_X and train_Y prior to training
	X , Y = process_train_data( X , Y )
	
	# train the model
	train_model(X , Y)





