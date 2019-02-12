import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.utils import to_categorical

############################################################
# Convolutional Network - TEST:
#  - Read training, validation and test data
#  - Train a convolutional network with one hidden layers
#  - Test data
############################################################


DATA_TRAIN = 'data/train.csv'
DATA_VALID = 'data/validation.csv'
DATA_TEST  = 'data/test.csv'


# Retrieve data from files
data_train = pd.read_csv(DATA_TRAIN, sep=',', header=None)
data_valid = pd.read_csv(DATA_VALID, sep=',', header=None)
data_test  = pd.read_csv(DATA_TEST,  sep=',', header=None)

# Separate datapoints (x) and label (y) from each row
def unpack_data(data):
    return (data.values[:,1:], data.values[:,0])

(x_train, y_train) = unpack_data(data_train)
(x_valid, y_valid) = unpack_data(data_valid)
(x_test,  y_test)  = unpack_data(data_test)

print()
print(" TRAINING  SAMPLES: ", len(x_train))
print("VALIDATION SAMPLES: ", len(x_valid))
print("   TEST    SAMPLES: ", len(x_test))

x_train = np.array(x_train).reshape(len(x_train),140,1).astype('float32')
x_valid = np.array(x_valid).reshape(len(x_valid),140,1).astype('float32')
x_test  = np.array(x_test ).reshape(len(x_test ),140,1).astype('float32')

y_test_raw  = y_test
y_train = np.array(to_categorical(y_train)).reshape(len(y_train),2).astype('int32')
y_valid = np.array(to_categorical(y_valid)).reshape(len(y_valid),2).astype('int32')
y_test  = np.array(to_categorical(y_test )).reshape(len(y_test ),2).astype('int32')

print()
print (" TRAINING  DATA SHAPE:\t", x_train.shape, "\t-> ", y_train.shape)
print ("VALIDATION DATA SHAPE:\t", x_valid.shape, "\t-> ", y_valid.shape)
print ("   TEST    DATA SHAPE:\t", x_test.shape, "\t-> ", y_test.shape)
print()




# Create model
model = Sequential()

# Add model layers
model.add(Conv1D(15, kernel_size=3, activation='relu', input_shape=(140,1)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30)



# Test on the test set
fx_test = model.predict(x_test)

# Let's say final solution is the obtained by considering the max between the 2 output neurons
max_fx_test = [0 if fx[0]>fx[1] else 1 for fx in fx_test]
positive_predictions = sum(1 for i, j in zip(max_fx_test, y_test_raw) if i == j)

print() 
print("TEST outcomes:")
print('\n'.join('{:4d}:  ({:1.2f}-{:1.2f}) --> prediction: {}\t{}'.format(i, fx[0], fx[1], max_fx_test[i], "YEAHS!" if max_fx_test[i]==y_test_raw[i] else "WRONG!") for i,fx in enumerate(fx_test)))
print()
print("Accuracy: ", float(positive_predictions)/len(x_test))




"""
Other stuff to try out(?)

model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())
"""