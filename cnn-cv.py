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
from tensorflow import set_random_seed

# Set seed for reproducible results
seed = 871
np.random.seed(seed)
set_random_seed(seed)

# Retrieve data from files
DATA = 'data/trainval-shifted.csv'
DATA_TRAIN = 'data/train-shifted.csv'
DATA_VAL = 'data/validation-shifted.csv'
DATA_TEST = 'data/test-shifted.csv'

def load_dataset(path):
    dataset = pd.read_csv(path, sep=',', header=None)
    X = dataset.values[:, 1:]
    Y = dataset.values[:, 0].astype('int32')
    X = X.reshape(len(X), 140, 1).astype('float32')
    return (X, Y)

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

def train_model(model, X_train, Y_train, X_val, Y_val):
    es = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=4,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, callbacks=[es])
    return model

def evaluate_model():
    print('\nEvaluating model...\n\n')
    (X, y) = load_dataset(DATA)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for i_train, i_val in kfold.split(X, y):
        model = build_model()

        # Train the model
        model = train_model(model, X[i_train], y[i_train], X[i_val], y[i_val])

        # Evaluate the model
        scores = model.evaluate(X[i_val], y[i_val], verbose =0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def test_model():
    print('\n\nTesting model...\n\n')
    model = build_model()

    (x_train, y_train) = load_dataset(DATA_TRAIN)
    (x_val, y_val) = load_dataset(DATA_VAL)
    (x_test, y_test) = load_dataset(DATA_TEST)

    # Train the model
    model = train_model(model, x_train, y_train, x_val, y_val)

    # evaluate the model
    _, acc = model.evaluate(x_test, y_test, verbose =0)
    print(acc)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            evaluate_model()
            sys.exit(0)

    # Default
    test_model()
