import sys
import pandas as pd
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

############################################################
# Convolutional Network - TEST:
#  - Read training, validation and test data
#  - Train a convolutional network with one hidden layers
#  - Test data
############################################################

seed = 7
np.random.seed(seed)

suffix = ''
if len(sys.argv) > 1:
    suffix = '-'+sys.argv[1]

# Retrieve data from files
DATA = 'data/trainval'+suffix+'.csv'
data_train = pd.read_csv(DATA, sep=',', header=None)

# Separate datapoints (x) and label (y) from each row
def unpack_data(data):
    return (data.values[:,1:], data.values[:,0])

(x_train, y_train) = unpack_data(data_train)

X = np.array(x_train).reshape(len(x_train),140,1).astype('float32')
y = np.array((y_train)).reshape(len(y_train),1).astype('int32')

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []


def build_model():
    # Create model
    model=Sequential()

    # Add model layers
    model.add(Conv1D(50, kernel_size=5, activation='relu', input_shape=(140,1)))
    model.add(Dropout(0.05))
    model.add(MaxPooling1D(10))
    model.add(Conv1D(25, kernel_size=3, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # lr is the learning rate of the optimizer
    optimizer = Adam(lr=0.001)

    # Compile model using accuracy to measure model performance
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model

for i_train, i_val in kfold.split(X, y):

    model = build_model()

    es = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=4,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    # Train the model
    model.fit(X[i_train], y[i_train], validation_data=(X[i_val], y[i_val]), epochs=100, callbacks=[es])
    # evaluate the model
    scores = model.evaluate(X[i_val], y[i_val], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
