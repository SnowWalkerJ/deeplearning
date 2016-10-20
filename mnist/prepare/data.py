import pandas as pd
import numpy as np
from tflearn.data_utils import to_categorical


trainset_csv = './data/train.csv'
trainset_npz = './data/train.npz'
testset_csv = './data/test.csv'
testset_npz = './data/test.npz'


def restore():
    restore_trainset()
    restore_testset()


def restore_testset():
    data = pd.read_csv(testset_csv)
    pixels = np.array(data, dtype=np.float64)
    pixels /= 255.0
    np.savez_compressed(testset_npz, pixels=pixels.reshape((-1, 28, 28, 1)))


def restore_trainset():
    data = pd.read_csv(trainset_csv)
    labels = np.array(data['label'])
    pixels = data[['pixel%d' % i for i in xrange(784)]]
    pixels = np.array(pixels, dtype=np.float64)
    pixels /= 255.0
    np.savez_compressed(trainset_npz, labels=labels, pixels=pixels.reshape((-1, 28, 28, 1)))


def load_trainset():
    data = np.load(trainset_npz)
    return to_categorical(data['labels'], 10), data['pixels']


def split_validation(dataset, rate):
    total = len(dataset)
    n = int(total * (1-rate))
    train = dataset[:n]
    valid = dataset[n:]
    return train, valid

def load_testset():
    return np.load(testset_npz)['pixels']


if __name__ == '__main__':
    restore()
