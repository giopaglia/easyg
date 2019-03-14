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

arg_num_neurons = 140
arg_num_neurons2 = 5
arg_rec_dropout = 0.0
arg_dropout = 0.1
arg_lr = 0.01
arg_n_splits = 14
arg_patience = 12
arg_batch_size = 20

if len(sys.argv) == 8:
    suffix = '-'+sys.argv[1]
    arg_num_neurons = int(sys.argv[2])
    arg_dropout     = float(sys.argv[3])
    arg_lr          = float(sys.argv[4])
    arg_n_splits    = int(sys.argv[5])
    arg_patience    = int(sys.argv[6])
    arg_batch_size  = int(sys.argv[7])

if len(sys.argv) == 10:
    suffix = '-'+sys.argv[1]
    arg_num_neurons = int(sys.argv[2])
    arg_num_neurons2= int(sys.argv[3])
    arg_rec_dropout = float(sys.argv[4])
    arg_dropout     = float(sys.argv[5])
    arg_lr          = float(sys.argv[6])
    arg_n_splits    = int(sys.argv[7])
    arg_patience    = int(sys.argv[8])
    arg_batch_size  = int(sys.argv[9])

DATA = 'data/trainval'+suffix+'.csv'

def load_dataset(path):
    dataset = pd.read_csv(path, sep=',', header=None)
    X = dataset.values[:, 1:]
    Y = dataset.values[:, 0].astype('int32')
    X =  X.reshape(X.shape[0], 1, X.shape[1])
    return (X, Y)

def build_model():
    model = Sequential()
    model.add(LSTM(arg_num_neurons, input_shape=(None, 140), recurrent_dropout=arg_rec_dropout))
    model.add(Dropout(arg_dropout))
    if (arg_num_neurons2 > 0): model.add(Dense(arg_num_neurons2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=arg_lr)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    return model

(X, y) = load_dataset(DATA)

kfold = StratifiedKFold(n_splits=arg_n_splits, shuffle=True, random_state=seed)
cvscores = []
for i_train, i_val in kfold.split(X, y):
    model = build_model()

    es = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=arg_patience,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    model.fit(X[i_train], y[i_train], validation_data=(X[i_val], y[i_val]), epochs=60, batch_size=arg_batch_size, callbacks=[es])

    # evaluate the model
    scores = model.evaluate(X[i_val], y[i_val], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
