from keras.models import Sequential
from keras.layers import Dense, Activation , Dropout
from keras.layers import LSTM

# Global Variables
LSTM_DIM_1 = 100
LSTM_DIM_2 = 200
DENSE_DIM_1 = 400
OUT_CLASS = 2


'''
function to design the model. 
LSTM based model with 2 LSTM layers and 1 Dense layer and 1 output layer
'''
def create_model(dim):
	model = Sequential()
	model.add(LSTM(LSTM_DIM_1 , return_sequences=True , input_shape=(7 , 300)))
	model.add(Activation('sigmoid'))
	
	model.add(LSTM(LSTM_DIM_2 , return_sequences = False))
	model.add(Activation('sigmoid'))

	model.add(Dense(DENSE_DIM_1))
	model.add(Activation('sigmoid'))
	
	model.add(Dense(OUT_CLASS))
	model.add(Activation('softmax'))
	return model


if __name__ == "__main__":
	create_model((7 , 300))
