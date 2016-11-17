from tf_learn.utils.data_utils import NpzData, LazyImageData, DataSet
import numpy as np


class Pipe(DataSet):
    def __init__(self, dataset, func):
        self.dataset = dataset

    @@property
    def length(self):
        return self.dataset.length

    def iterate(self, batch_size=1):
        for label, image in self.dataset.iterate(1):
            yield label, image
            yield label, np.asarray(map(lambda x: np.asarray(list(reversed(x))), image[0]))



data = LazyImageData('data/train', lambda name: 1 if name.startswith('cat') else 0)
# data = NpzData('dump/data.npz')
# images.set_resize((300, 300))
data.split([0.99])
NpzData.dump(data[1], 'dump/validate.npz')
piped = Pipe(data[0])
NpzData.dump(piped, 'dump/train.npz')

