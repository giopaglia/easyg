import sys
import numpy as np
import pandas as pd
import sklearn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

seed = 7

suffix = '-shifted'
if len(sys.argv) == 2:
    suffix = '-'+sys.argv[1]

arg_num_neurons = 100
arg_lr = 0.001
arg_n_splits = 10
arg_patience = 2
arg_batch_size = 50

if len(sys.argv) == 7:
    suffix = '-'+sys.argv[1]
    arg_num_neurons = int(sys.argv[2])
    arg_lr          = float(sys.argv[3])
    arg_n_splits    = int(sys.argv[4])
    arg_patience    = int(sys.argv[5])
    arg_batch_size  = int(sys.argv[6])

DATA = 'data/trainval'+suffix+'.csv'

def load_dataset(path):
    dataset = pd.read_csv(path, sep=',', header=None)
    X = dataset.values[:, 1:]
    Y = dataset.values[:, 0].astype('int32')
    X =  X.reshape(X.shape[0], 1, X.shape[1])
    return (X, Y)

def build_model():
    model = Sequential()
    model.add(LSTM(arg_num_neurons, input_shape=(None, 140)))
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

    model.fit(X[i_train], y[i_train], validation_data=(X[i_val], y[i_val]), epochs=30, batch_size=arg_batch_size, callbacks=[es])

    # evaluate the model
    scores = model.evaluate(X[i_val], y[i_val], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
