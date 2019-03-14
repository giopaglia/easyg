import sys
import numpy as np
import pandas as pd
import sklearn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold


# the random seed initialization in tensor flow is sufficient to get reproducible results

#######################################
#import random
#import time
# seed = random.seed(time.time())
#######################################
#from numpy.random import seed
#seed(1)
#######################################
#from tensorflow import set_random_seed    #  <- This
#set_random_seed(2)                        #   <- one
#######################################
seed = 7
#######################################



suffix = '-shifted'
#suffix = ''
if len(sys.argv) == 2:
    suffix = '-'+sys.argv[1]

DATA = 'data/trainval'+suffix+'.csv'

def load_dataset(path):
    dataset = pd.read_csv(path, sep=',', header=None)
    X = dataset.values[:, 1:]
    Y = dataset.values[:, 0].astype('int32')
    X =  X.reshape(X.shape[0], 1, X.shape[1])
    return (X, Y)

def build_model():
    model = Sequential()
    model.add(LSTM(140, input_shape=(None, 140)))
    model.add(Dropout(0.1))
    model.add(Dense(5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.01)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    return model

(X, y) = load_dataset(DATA)

kfold = StratifiedKFold(n_splits=14, shuffle=True, random_state=seed)
cvscores = []
for i_train, i_val in kfold.split(X, y):
    model = build_model()

    es = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=12,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    model.fit(X[i_train], y[i_train], validation_data=(X[i_val], y[i_val]), epochs=60, batch_size=20, callbacks=[es])

    # evaluate the model
    scores = model.evaluate(X[i_val], y[i_val], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
