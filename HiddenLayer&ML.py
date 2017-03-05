import numpy as np
np.random.seed(1337)  # for reproducibility
import os
os.environ['THEANO_FLAGS'] = "device=gpu"    #Use GPU for calculation
import theano

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, GaussianNoise
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import SimpleRNN, Activation
from keras.regularizers import l2, activity_l2

TIME_STEPS = 28     # same as the height of the image
INPUT_SIZE = 28     # same as the width of the image
BATCH_SIZE = 5000
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# build RNN model
model = Sequential()

# RNN cell
size = X_train.shape
model = Sequential()
model.add(LSTM(batch_input_shape=(None,1, 72),output_dim=100,return_sequences=True,))
model.add(LSTM(output_dim=50,return_sequences=True,))
model.add(LSTM(output_dim=10,return_sequences=False,))
model.add(GaussianNoise(1))
model.add(Dropout(0.5)) #dropout to avoid overfitting
model.add(Dense(2)) #Because we have 4 kind of output
model.add(Activation('sigmoid'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.summary()

X_train = X_train.reshape(-1,1,72)
X_test = X_test.reshape(-1,1,72)
Y_train_1 = np_utils.to_categorical(Y_train, nb_classes=2)
Y_test_1 = np_utils.to_categorical(Y_test, nb_classes=2)

#because we store our data in HDF5 file, so only weights are stored, we need to redefine our model first before loading weights
model.load_weights('modelfrom400to900.h5')
#If our model don't have any dropout or batch normalization, we can use the following to get the intermidiate layer
#Previously trained and save model is named 'model'
intermediate_layer_model = model(input=model.input,output=model.get_layer(model.layers[2].name).output)
intermediate_output = intermediate_layer_model.predict(data)

#Our model contains dropout, so we should use the following method
from keras import backend as K
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[4].output])

# output in test mode = 0
layer_output_test = get_3rd_layer_output([X_test, 0])[0]

# output in train mode = 1
layer_output_train = get_3rd_layer_output([X_train, 1])[0]

#Get the training accuracy when we combine output of LSTM and random forest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
X_train_ML = layer_output_train.reshape(-1,10)
Y_train_ML = Y_train.reshape(-1,)
RFLearner1 = RandomForestClassifier()
RFLearner1.fit(X_train_ML,Y_train_ML)
RFLearner1.score(X_train_ML,Y_train_ML)

#Here we test the accuracy for test set
X_test_ML = layer_output_test.reshape(-1,10)
Y_test_ML = Y_test.reshape(-1,)
RFLearner1.score(X_test_ML,Y_test_ML)
