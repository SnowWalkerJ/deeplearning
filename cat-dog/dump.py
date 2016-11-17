from tf_learn.utils.data_utils import NpzData, LazyImageData, DataSet
import numpy as np


class Pipe(DataSet):
    def __init__(self, dataset, func):
        self.dataset = dataset
        self.func = func

    @property
    def length(self):
        return self.dataset.length * 2

    def iterate(self, batch_size=1):
        for label, image in self.dataset.iterate(1):
            yield label, image
            yield label, self.func(image)


data = LazyImageData('data/train', lambda name: 1 if name.startswith('cat') else 0)
# data = NpzData('dump/data.npz')
data.set_resize((300, 300))
data.split([0.99])
NpzData.dump(data[1], 'dump/validate.npz')
piped = Pipe(data[0], np.fliplr)
NpzData.dump(piped, 'dump/train.npz')

