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
from tensorflow import set_random_seed

# Set seed for reproducible results
seed = 871
np.random.seed(seed)
set_random_seed(seed)

DATA        = 'data/trainval-shifted.csv'
DATA_TRAIN  = 'data/train-shifted.csv'
DATA_VAL    = 'data/validation-shifted.csv'
DATA_TEST   = 'data/test-shifted.csv'

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

    # lr is the learning rate of the optimizer
    optimizer = Adam(lr=0.001)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    es = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=12,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=60, batch_size=20, callbacks=[es])
    return model

def evaluate_model():
    print('\n\Evaluating model...\n\n')
    cvscores = []
    (X, y) = load_dataset(DATA)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for i_train, i_val in kfold.split(X, y):
        model = build_model()

        # Train the model
        model = train_model(model, X[i_train], y[i_train], X[i_val], y[i_val])

        # Evaluate the model
        scores = model.evaluate(X[i_val], y[i_val], verbose =0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def test_model():
    print('\n\nTesting model...\n\n')
    model = build_model()

    (x_train, y_train) = load_dataset(DATA_TRAIN)
    (x_val, y_val) = load_dataset(DATA_VAL)
    (x_test, y_test) = load_dataset(DATA_TEST)

    # Train the model
    model = train_model(model, x_train, y_train, x_val, y_val)

    # Evaluate the model
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(acc)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            evaluate_model()
            sys.exit(0)

    # Default
    test_model()
