from prepare.data import load_trainset, split_validation
from manipulate import generate_new_data
from net import build_net
import tensorflow as tf
import numpy as np
# import tflearn.datasets.mnist as mnist


if __name__ == '__main__':
    labels, pixels = load_trainset()
    labels_train, labels_valid = split_validation(labels, 0.01)
    pixels_train, pixels_valid = split_validation(pixels, 0.01)

    for s in xrange(5):
        print "generating data: %d / %d" % (s+1, 5)
        labels_extend, pixels_extend = generate_new_data(labels_train, pixels_train, 5000, probs=(0.3, 0.3, 0.3))
        pixels_train = np.append(pixels_train, pixels_extend, 0)
        labels_train = np.append(labels_train, labels_extend, 0)
    print "data generating complete."

    with tf.Session() as session:
        model = build_net()
        model.train(pixels_train, labels_train, valid_x=pixels_valid, valid_y=labels_valid, n_epoch=20,
                    params=dict(keep_prob=0.8), eval_params=dict(keep_prob=1.0), nsleep=10)

    # print labels.shape, pixels.shape
    
