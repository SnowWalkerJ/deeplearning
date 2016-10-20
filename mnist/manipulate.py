"""
This module helps create new dataset by adding salt, torture, averaging old pictures
"""
import random
import numpy as np
from PIL import Image
from tflearn.data_utils import to_categorical


def average(pic1, pic2):
    """
    if pic1 and pic2 are pictures of the same class, then their average should be of that class too
    """
    return (pic1 + pic2) / 2.0


def salt(pic, prob=0.05, strength=0.5):
    """
    add noise points to the picture
    :param pic:
    :param strength:
    :return:
    """
    origin_size = pic.shape
    noise = np.array([(random.random()*strength*2-strength) if random.random() < prob else 0
                      for _ in xrange(pic.size)])\
        .reshape(origin_size)
    return pic + noise


def torture(pic, prob=0.1):
    origin_size = pic.shape
    pic2 = pic.reshape(origin_size[:2])
    for row in xrange(origin_size[0]):
        if random.random() > prob:
            continue
        if random.random() > 0.5:
            pic2[row] = [0] + list(pic2[row, :-1])
        else:
            pic2[row] = list(pic2[row, 1:]) + [0]
    return pic2.reshape(origin_size)


def rotate(pic, degree):
    origin_size = pic.shape
    pic2 = Image.fromarray(pic.reshape(origin_size[:2])*255)
    pic2.rotate(degree)
    return np.array(pic2.getdata()).reshape(origin_size) / 255.0


def shift(pic, x_offset, y_offset):
    """
    No need for this kind of manipulation because of the CNN
    """
    origin_size = pic.size
    pic2 = Image.fromarray(pic.reshape(origin_size[:2])*255, 'L')
    pic2.offset(x_offset, y_offset)
    return np.array(pic2.getdata()).reshape(origin_size) / 255.0


def generate_new_data(labels, pixels, number, probs=(0.4, 0.5, 0.6)):
    indice = range(len(labels))
    new_data = []
    new_labels = []
    for i in xrange(number):
        dice = random.random()
        if dice < probs[0]:
            _class = random.randint(0, 9)
            subset = pixels[labels[:, _class] == 1]
            # average
            pic1, pic2 = random.choice(subset), random.choice(subset)
            _class = to_categorical([_class], 10)[0]
            new_pic = average(pic1, pic2)
        elif dice < probs[1]:
            # salt
            j = random.choice(indice)
            _class = labels[j]
            new_pic = salt(pixels[j], prob=0.2, strength=random.random()*0.7)
        elif dice < probs[2]:
            # torture
            j = random.choice(indice)
            _class = labels[j]
            new_pic = torture(pixels[j], prob=0.2)
        else:
            # rotate
            j = random.choice(indice)
            _class = labels[j]
            new_pic = rotate(pixels[j], random.randint(-25, 25))
        assert new_pic.shape == (28, 28, 1), dice
        new_data.append(new_pic)
        new_labels.append(_class)
    return np.array(new_labels), np.array(new_data)


if __name__ == '__main__':
    from PIL import Image
    from prepare.data import load_trainset
    labels, pixels = load_trainset()
    indice = range(len(labels))
    j = random.choice(indice)
    _class = labels[j]
    new_pic = rotate(pixels[j], random.randint(-25, 25)).reshape((28, 28)) * 255
    print new_pic
    print _class
    Image.fromarray(new_pic).show()
