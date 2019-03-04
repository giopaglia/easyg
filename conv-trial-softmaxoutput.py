import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.utils import to_categorical

############################################################
# Convolutional Network - TEST:
#  - Read training and validation data
#  - Train a convolutional network with one hidden layers
############################################################

suffix = ''
if len(sys.argv) > 1:
    suffix = '-'+sys.argv[1]


DATA_TRAIN = 'data/train'+suffix+'.csv'
DATA_VALID = 'data/validation'+suffix+'.csv'


# Retrieve data from files
data_train = pd.read_csv(DATA_TRAIN, sep=',', header=None)
data_valid = pd.read_csv(DATA_VALID, sep=',', header=None)

# Separate datapoints (x) and label (y) from each row
def unpack_data(data):
    return (data.values[:,1:], data.values[:,0])

(x_train, y_train) = unpack_data(data_train)
(x_valid, y_valid) = unpack_data(data_valid)

print()
print(" TRAINING  SAMPLES: ", len(x_train))
print("VALIDATION SAMPLES: ", len(x_valid))

x_train = np.array(x_train).reshape(len(x_train),140,1).astype('float32')
x_valid = np.array(x_valid).reshape(len(x_valid),140,1).astype('float32')

y_valid_raw  = y_valid
y_train = np.array(to_categorical(y_train)).reshape(len(y_train),2).astype('int32')
y_valid = np.array(to_categorical(y_valid)).reshape(len(y_valid),2).astype('int32')

print()
print (" TRAINING  DATA SHAPE:\t", x_train.shape, "\t-> ", y_train.shape)
print ("VALIDATION DATA SHAPE:\t", x_valid.shape, "\t-> ", y_valid.shape)
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



# Test on the validation set
_, acc = model.evaluate(x_valid, y_valid)
print('Accuracy: {0}'.format(acc * 100))
