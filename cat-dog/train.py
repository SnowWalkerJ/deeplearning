from tf_learn.utils.data_utils import LazyImageData, BigData
from net import Model
import os


logdir = 'logs'

if __name__ == '__main__':
    data = BigData('dump')
    # data = LazyImageData('data/train', lambda name: 1 if name.split('.')[0] == 'cat' else 0)
    # data.set_crop((300, 300))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    model = Model(logdir)
    placeholders = {
        'keep_prob': {
            'train': 0.5,
            'evaluate': 1.0,
        },
    }
    model.train(data, batch_size=64, placeholders=placeholders, validate=0.01, save_name='saves/model1')

