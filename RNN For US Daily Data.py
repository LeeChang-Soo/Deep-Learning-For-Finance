import os
import datetime as datetime
import glob
import pandas as pd
import csv
import numpy as np
import operator
import tushare as ts
#print(os.getcwd())
os.chdir('/Users/sunjiaxuan/Downloads/quantquote_daily_sp500_83986/daily')
AllFiles = glob.glob("/Users/sunjiaxuan/Downloads/quantquote_daily_sp500_83986/daily/*.csv")

#首先读入所有历史数据，然后计算其daily和monthly的return
#然后得到y，再进行预测，训练各种区分方
len(AllFiles)
FileNames = []
FileResult =[]
AllReturn = [] #Open price change
AllVolChange = []
AllOscilation = []
AllDates = []
#Problem here is that we have different date length for different stocks
#The solution we want
for files in AllFiles:
    if 'table' in files:
        FileNames.append(files)
        temp = pd.read_csv(files,header=None)
        temp.columns = ['Date','X','Open','High','Low','Close','Vol']
        FileResult.append(temp)
        AllReturn.append(np.log(temp['Open'] / temp['Open'].shift(1)))
        AllVolChange.append(temp['Vol'] / temp['Vol'].shift(1))
        AllOscilation.append((temp['High']-temp['Low'])/temp['Open'])
        AllDates.append(temp['Date'])
        if files == AllFiles[0]:
            RetDF = pd.DataFrame(np.log(temp['Open'] / temp['Open'].shift(1)).tolist(),index=temp['Date'])
            VolDF = pd.DataFrame((temp['Vol'] / temp['Vol'].shift(1)).tolist(),index=temp['Date'])
            OscDF = pd.DataFrame(((temp['High']-temp['Low'])/temp['Open']).tolist(),index=temp['Date'])
        else:
            RetDF = pd.concat([RetDF,pd.DataFrame(np.log(temp['Open'] / temp['Open'].shift(1)).tolist(),index=temp['Date'])],axis=1)
            VolDF = pd.concat([VolDF,pd.DataFrame((temp['Vol'] / temp['Vol'].shift(1)).tolist(),index=temp['Date'])],axis=1)
            OscDF = pd.concat([OscDF,pd.DataFrame(((temp['High']-temp['Low'])/temp['Open']).tolist(),index=temp['Date'])], axis=1)

#We can try 2 different structures here, one is using all historical data for 20 periods ahead and using RBM to get features (we can also try this method in Ricequant)
#We can also try to use RNN, with out using RBM (We can also try that)

#Step 1, get the mean returns in each period and change column names for later calculation
MeanRet = RetDF.mean(axis = 1)
MeanVol = VolDF.mean(axis = 1) - 1 #Because we want to get a change or not
VolDF = VolDF - 1 #We need to make the VolDF as a rate of change
MeanOsc = OscDF.mean(axis = 1)
RetDF.columns = list(range(0,500))
VolDF.columns = list(range(0,500))
OscDF.columns = list(range(0,500))
#Try Version 1, Use RBM and other classifiers
#S1, prepate data
X = np.array([]) #Every training example contains 12 monthly return, 8 weekly return and 20 Vol changes and 20 OsiChange
Y = np.array([])
size = RetDF.shape #Here these 3 dataframes have same length in dates
start = 3000 #Could be 300
end = 3200 #Could be size[0], the recent performance is also good
#Also, we need to make sure that MeanRet is in the same order as tempRet, we adjust for market performance later
def GetMonthlyReturn(tempRet):
    if len(tempRet) != 240:
        print('Wrong Input for Monthly Return')
        return
    else:
        Re = np.array([])
        MonthReturn = 0
        for i in list(range(0,240)):
            MonthReturn = MonthReturn + tempRet[i]
            if i%20 == 19:
                Re = np.append(Re,MonthReturn)
                MonthReturn = 0
        return Re

#Make sure we all use array here, we can combine them first and them reshape them
#Here we change the input data to relative to market ones
for i in list(range(start,end)):#When we change 500 to size[0]-20,because we still need some time to compute the y
    print(i)
    for j in list(range(0,500)):
        #print(j)
        if (i == start)&(j==0):
            tempRet = RetDF.iloc[(i-240):i,j].values
            tempMonthlyReturn = GetMonthlyReturn(tempRet)
            tempDailyReturn = RetDF.iloc[(i-20):i,j].values
        else:
            tempRet = RetDF.iloc[(i-240):i,j].values
            tempMonthlyReturn = np.append(tempMonthlyReturn,GetMonthlyReturn(tempRet))
            tempDailyReturn = np.append(tempDailyReturn,RetDF.iloc[(i-20):i,j].values)

    tempMonthlyReturn = tempMonthlyReturn.reshape(-1,12)
    tempMonthlyMean = np.nanmean(tempMonthlyReturn,axis = 0)
    tempDailyReturn = tempDailyReturn.reshape(-1, 20)
    tempDailyMean = np.nanmean(tempDailyReturn, axis=0)

    for j in list(range(0,500)):
        #print(j)
        if (i == start)&(j==0):
            tempRet = RetDF.iloc[(i-240):i,j].values
            MonthlyReturn = GetMonthlyReturn(tempRet) - tempMonthlyMean
            DailyReturn = RetDF.iloc[(i-20):i,j].values - tempDailyMean
            VolChange = VolDF.iloc[(i-20):i,j].values
            Osc = OscDF.iloc[(i-20):i,j].values
        else:
            tempRet = RetDF.iloc[(i-240):i,j].values
            MonthlyReturn = np.append(MonthlyReturn,GetMonthlyReturn(tempRet) - tempMonthlyMean)
            DailyReturn = np.append(DailyReturn,RetDF.iloc[(i-20):i,j].values - tempDailyMean)
            VolChange = np.append(VolChange,VolDF.iloc[(i-20):i,j].values)
            Osc = np.append(Osc,OscDF.iloc[(i-20):i,j].values)

MonthlyReturn = MonthlyReturn.reshape(-1,12)
DailyReturn = DailyReturn.reshape(-1,20)
VolChange = VolChange.reshape(-1,20)
Osc = Osc.reshape(-1,20)

AllX = np.concatenate((MonthlyReturn,DailyReturn,VolChange,Osc),axis=1)
AllX.shape
AllY = np.array([])
for i in list(range(start,end)):
    print(i)
    TodayReturn = np.array([])
    for j in list(range(0,500)):
        if (i == start)&(j==0):
            tempRet = RetDF.iloc[i:(i+20),j].values
            TodayReturn = np.append(TodayReturn,sum(tempRet))
        else:
            tempRet = RetDF.iloc[i:(i+20),j].values
            TodayReturn = np.append(TodayReturn, sum(tempRet))
    TodayMean = np.nanmean(TodayReturn)
    for j in list(range(0,500)):
        if (i == start)&(j==0):
            tempRet = RetDF.iloc[i:(i+20),j].values
            AllY = sum(tempRet) - TodayMean
        else:
            tempRet = RetDF.iloc[i:(i+20),j].values
            AllY = np.append(AllY,sum(tempRet) - TodayMean)

AllY.shape
temp = AllY.copy()
temp[AllY>0] = 1
temp[AllY<0] = 0
AllY = temp

#Deal with nan value here
#tempa = np.concatenate((AllX,AllY),axis = 1)
tempDFX = pd.DataFrame(AllX)
tempDFY = pd.DataFrame(AllY)
tempDF = pd.concat((tempDFX,tempDFY),axis=1)
tempDF = tempDF.dropna()
tempDF.to_csv("/Users/sunjiaxuan/Documents/Py/YahooAnalysts/DataForTraining2000to2400.csv")
tempDF = pd.read_csv("/Users/sunjiaxuan/Documents/Py/YahooAnalysts/DataForTraining2000to2400.csv")
tempDF = tempDF.iloc[:,1:]
AllX = tempDF.iloc[:,0:-1].values
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


print("LR using raw features in sample:\n%s\n" % (
    metrics.classification_report(
        Y_train,
        logistic_classifier.predict(X_train))))

print("SVM using raw features out of sample:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        clf.predict(X_test))))

print("SVM using raw features in sample:\n%s\n" % (
    metrics.classification_report(
        Y_train,
        clf.predict(X_train))))

######################We can also try RNN Here
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
BATCH_SIZE = 8000
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# build RNN model
model = Sequential()

# RNN cell
size = X_train.shape
model = Sequential()
model.add(LSTM(batch_input_shape=(None,1, 72),output_dim=50,return_sequences=True,))
#model.add(LSTM(output_dim=50,return_sequences=True,))
model.add(LSTM(output_dim=10,return_sequences=False,))
model.add(GaussianNoise(1))
model.add(Dropout(0.5)) #dropout,两层LSTM和Dropout效果比较好
model.add(Dense(2)) #Because we have 4 kind of output
model.add(Activation('sigmoid'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.summary()

X_train = X_train.reshape(-1,1,72)
X_test = X_test.reshape(-1,1,72)
Y_train_1 = np_utils.to_categorical(Y_train, nb_classes=2)
Y_test_1 = np_utils.to_categorical(Y_test, nb_classes=2)
# model.fit(X_train,Y_train_1)
# training
for step in range(40001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = Y_train_1[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE,:]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 50 == 0:
        cost, accuracy = model.evaluate(X_test, Y_test_1, batch_size=Y_test.shape[0], verbose=False)
        cost1, accuracy1 = model.evaluate(X_batch, Y_batch, batch_size=Y_batch.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
        print('train cost: ', cost1, 'train accuracy: ', accuracy1)

###Save result model to file
model.save('modelfrom2000to2400.h5')   # HDF5 file, you have to pip3 install h5py if don't have it

#################Use a more sopiscated model
temp1 = pd.read_csv("/Users/sunjiaxuan/Documents/Py/YahooAnalysts/DataForTraining400to900.csv")
temp2 = pd.read_csv("/Users/sunjiaxuan/Documents/Py/YahooAnalysts/DataForTraining2000to2400.csv")
All = pd.concat((temp1,temp2),axis=0)
All = All.iloc[:,1:] #Because indexes are read in as values
AllX = All.iloc[:,0:-1].values
AllY = All.iloc[:,-1].values
#Normalize AllX
AllX = (AllX - np.min(AllX, 0)) / (np.max(AllX, 0) + 0.0001)  # 0-1 scaling

from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

X_train, X_test, Y_train, Y_test = train_test_split(AllX, AllY,
                                                    test_size=0.2,
                                                    random_state=0)

# build RNN model
model_New = Sequential()

# RNN cell
size = X_train.shape
model_New = Sequential()
model_New.add(LSTM(batch_input_shape=(None,1, 72),output_dim=30,return_sequences=True,))
#model_New.add(LSTM(output_dim=30,return_sequences=True,))
model_New.add(LSTM(output_dim=10,return_sequences=False,))
model_New.add(GaussianNoise(1))
model_New.add(Dropout(0.5)) #dropout,两层LSTM和Dropout效果比较好
model_New.add(Dense(2)) #Because we have 4 kind of output
model_New.add(Activation('sigmoid'))
model_New.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model_New.summary()

X_train = X_train.reshape(-1,1,72)
X_test = X_test.reshape(-1,1,72)
Y_train_1 = np_utils.to_categorical(Y_train, nb_classes=2)
Y_test_1 = np_utils.to_categorical(Y_test, nb_classes=2)
#model.fit(X_train,Y_train_1)
# training 需要非常多次才收敛
for step in range(12001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = Y_train_1[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE,:]
    cost = model_New.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 50 == 0:
        cost, accuracy = model_New.evaluate(X_test, Y_test_1, batch_size=Y_test.shape[0], verbose=False)
        cost1, accuracy1 = model_New.evaluate(X_batch, Y_batch, batch_size=Y_batch.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
        print('train cost: ', cost1, 'train accuracy: ', accuracy1)

#####Try TA-Lib to generate techinical indicators to use as input for behavior predication
