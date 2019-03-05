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
np.random.seed(seed)

suffix = ''
if len(sys.argv) > 1:
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
    model.add(LSTM(100, input_shape=(None, 140)))
    model.add(Dense(1, activation='sigmoid'))

    # lr is the learning rate of the optimizer
    optimizer = Adam(lr=0.001)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    return model

(X, y) = load_dataset(DATA)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for i_train, i_val in kfold.split(X, y):
    model = build_model()

    es = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=2,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    model.fit(X[i_train], y[i_train], validation_data=(X[i_val], y[i_val]), epochs=30, batch_size=50, callbacks=[es])

    # evaluate the model
    scores = model.evaluate(X[i_val], y[i_val], verbose =0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
