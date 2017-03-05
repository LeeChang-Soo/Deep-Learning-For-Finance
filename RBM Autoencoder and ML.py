#####################################################
##Do autocoder and decoder for feature generation, seems not generating promising result
from keras.models import Model
from keras.layers import Dense, Input
X_train = X_train.reshape(-1,72)
X_test = X_test.reshape(-1,72)
# this is our input placeholder
input_feature = Input(shape=(72,))
encoded = Dense(40, activation='relu')(input_feature)
encoded = Dense(20, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(5)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(20, activation='relu')(decoded)
decoded = Dense(40, activation='relu')(decoded)
decoded = Dense(72, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_feature, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_feature, output=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()
# training
autoencoder.fit(X_train, X_train,
                nb_epoch=20,
                batch_size=1000,
                shuffle=True)

X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train_encoded, Y_train)

# Training SVM
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train_encoded, Y_train)

print()
print("Logistic regression using encoded features:\n%s\n" % ( #seems not working well in this sample
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test_encoded))))

print("LR using raw features in sample:\n%s\n" % (
    metrics.classification_report(
        Y_train,
        logistic_classifier.predict(X_train))))

print("SVM using raw features out of sample:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        clf.predict(X_test_encoded))))

print("SVM using raw features in sample:\n%s\n" % (
    metrics.classification_report(
        Y_train,
        clf.predict(X_train_encoded))))
        
        
AllY = tempDF.iloc[:,-1].values
#Normalize AllX
AllX = (AllX - np.min(AllX, 0)) / (np.max(AllX, 0) + 0.0001)  # 0-1 scaling

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

X_train, X_test, Y_train, Y_test = train_test_split(AllX, AllY,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

rbm.learning_rate = 0.01
rbm.n_iter = 50
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 10
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

# Training SVM
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, Y_train)

print()
print("Logistic regression using RBM features:\n%s\n" % ( #seems not converging
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))
