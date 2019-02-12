import pandas as pd
import numpy as np

seed = 58123
np.random.seed(seed)

DATA_TEST = 'data/ECG5000_TEST.txt'
DATA_TRAIN = 'data/ECG5000_TRAIN.txt'

def filter_condition(row):
    return row[0] == 1 or row[0] == 2

def concat_sources():
    data_1 = pd.read_csv(DATA_TEST, sep=',', header=None)
    data_2 = pd.read_csv(DATA_TRAIN, sep=',', header=None)
    return np.concatenate((data_1.values, data_2.values))

def filter_data(data, condition):
    return data[np.array([condition(row) for row in data])]

def save(data, path):
    pd.DataFrame(data).to_csv(path, header=None, index=None)
    print('Saved {0} data points to {1}'.format(len(data), path))

def split_data(data, r_train=0.6, r_val=0.3, r_test=0.1):
    if np.round(r_train + r_val + r_test) != 1.0:
        raise Exception("Ratios should sum to 1.0")
    l = len(data)
    idx_train = int(np.floor(l * r_train))
    idx_val = int(np.floor(l * r_val) + idx_train + 1)
    data_train = data[:idx_train]
    data_val = data[idx_train:idx_val]
    data_test = data[idx_val:]
    return data_train, data_val, data_test

def main():
    filtered = filter_data(concat_sources(), filter_condition)
    np.random.shuffle(filtered)
    train, val, test = split_data(filtered)
    save(train, 'data/train.csv')
    save(val, 'data/validation.csv')
    save(test, 'data/test.csv')

if __name__ == "__main__":
    main()
