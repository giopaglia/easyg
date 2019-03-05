import sys
import pandas as pd
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

############################################################
# Convolutional Network - TEST:
#  - Read training, validation and test data
#  - Train a convolutional network with one hidden layers
#  - Test data
############################################################

seed =7

suffix = ''
if len(sys.argv) > 1:
    suffix = '-'+sys.argv[1]


DATA_TRAIN = 'data/trainval'+suffix+'.csv'
#DATA_VALID = 'data/validation.csv'
#DATA_TEST  = 'data/test.csv'


# Retrieve data from files
data_train = pd.read_csv(DATA_TRAIN, sep=',', header=None)
#data_valid = pd.read_csv(DATA_VALID, sep=',', header=None)
#data_test  = pd.read_csv(DATA_TEST,  sep=',', header=None)

# Separate datapoints (x) and label (y) from each row
def unpack_data(data):
    return (data.values[:,1:], data.values[:,0])

(x_train, y_train) = unpack_data(data_train)
#(x_valid, y_valid) = unpack_data(data_valid)
#(x_test,  y_test)  = unpack_data(data_test)

print()
print(" TRAINING  SAMPLES: ", len(x_train))
#print("VALIDATION SAMPLES: ", len(x_valid))
#print("   TEST    SAMPLES: ", len(x_test))

x_train = np.array(x_train).reshape(len(x_train),140,1).astype('float32')
#x_valid = np.array(x_valid).reshape(len(x_valid),140,1).astype('float32')
#x_test  = np.array(x_test ).reshape(len(x_test ),140,1).astype('float32')

#y_valid_raw  = y_valid
y_train = np.array((y_train)).reshape(len(y_train),1).astype('int32') # y_train = np.array(to_categorical(y_train)).reshape(len(y_train),2).astype('int32')
#y_valid = np.array((y_valid)).reshape(len(y_valid),1).astype('int32') # y_valid = np.array(to_categorical(y_valid)).reshape(len(y_valid),2).astype('int32')
#y_test  = np.array((y_test )).reshape(len(y_test ),1).astype('int32') # y_test  = np.array(to_categorical(y_test )).reshape(len(y_test ),2).astype('int32')

print()
print (" TRAINING  DATA SHAPE:\t", x_train.shape, "\t-> ", y_train.shape)
#print ("VALIDATION DATA SHAPE:\t", x_valid.shape, "\t-> ", y_valid.shape)
#print ("   TEST    DATA SHAPE:\t", x_test.shape, "\t-> ", y_test.shape)
print()
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x_train, y_train):

    # Create model
    model=Sequential()

    # Add model layers
    model.add(Conv1D(15, kernel_size=3, activation='relu', input_shape=(140,1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train[train], y_train[train], validation_data=(x_train[test], y_train[test]), epochs=30)
    # evaluate the model
    scores = model.evaluate(x_train[test], y_train[test], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
