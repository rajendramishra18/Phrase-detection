import cPickle as cp
import numpy as np
from sentence_matrix import prepare_sentence_matrix
from main import create_chunks
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint , EarlyStopping , History 
from create_network import create_model
from keras.optimizers import SGD , Adam
from keras.models import load_model


''' Function to train model '''
def train_model(train_X , train_Y):
	
	# We will store only the best possible model based on the val_loss. 
	# In turn it helps us avoiding overfitting.
	checkpoint = ModelCheckpoint("model/weight_2.h5" ,  monitor='val_loss' , verbose = 1 , save_best_only=True)
	
	# EarlyStopping is again a measure to avoid overfitting.
	es = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
	
	# Create the model which is define in create_network. 
	# Input dimention is 7*300 i.e. each chunk is sent to the model for training. 
	model = load_model('model/weight_1.h5')
	
	print model.summary()
	
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	
	#~ # Compile the model with optimizer 'Adam' and loss 'binary_crossentropy'
	#~ model.compile(loss='binary_crossentropy', optimizer='Adam' , metrics=['accuracy'])
	
	# save the model architecture
	json_string = model.to_json()
	fp = open("model/model_2_arch" , "w")
	fp.write(json_string)
	
	# train the model for 100 epochs with earlystopping and batch size 32
	model.fit(train_X , train_Y , batch_size = 32 , nb_epoch = 100 , verbose = 1 , callbacks = [checkpoint , es] , validation_split = .2 , shuffle = True , )
	
	
''' Function to convert the training lists to respective numpy arrays '''
def format_training_data(file_X_in):
	# load already prepared data for training 
	train_X = cp.load(open(file_X_in , 'r'))
	
	# Convert the lists to numpy array for training
	train_X = np.array(train_X)
	print(train_X.shape)
	
	# prepare train Y for not found case
	train_Y = np.zeros((len(train_X) , 30))
	print(train_Y.shape)
	# Print the sizes of the array and check if they are same
	np.save("train_X_nf_case.npy" , train_X)
	np.save("train_Y_nf_case.npy" , train_Y)
	return train_X , train_Y


if __name__ == "__main__":
	#~ prepare_sentence_matrix("not_found_sent_list.cp" , "train_X_final_not_found.cp")
	
	
	#~ train_X , train_Y = format_training_data("train_X_final_not_found.cp")
	
	# load numpy array containing the not found case sentences 
	train_X = np.load("train_X_nf_case.npy") 
	train_Y = np.load("train_Y_nf_case.npy") 
	
	# create chunks of window 7
	X , Y = create_chunks(train_X , train_Y , "index_len_list_NF.cp")
	
	# load the numpy arrays of normal sentences that was used to train first model
	chunk_X = np.load("chunk_X.npy")
	chunk_Y = np.load("chunk_y.npy")
	
	# print shape
	print(chunk_X.shape)

	# concatenate two numpy arrays : first the numpy array for normal case and second numpy array of not found case 
	X = np.concatenate((chunk_X, np.array(X)), axis=0)
	Y = np.concatenate((chunk_Y, np.array(Y)), axis=0)
	
	print X.shape
	print Y.shape
	
	# train the model_2 on combined data
	train_model(X , Y)




