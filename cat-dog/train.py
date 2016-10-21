from tf_learn.utils.data_utils import LazyImageData
from net import Model


if __name__ == '__main__':
    data = LazyImageData('data/train', lambda name: 1 if name.split('.')[0] == 'cat' else 0)
    data.set_crop((300, 300))
    model = Model()
    placeholders = {
        'keep_prob': {
            'train': 0.5,
            'evaluate': 1.0,
        },
    }
    model.train(data, batch_size=32, placeholders=placeholders, validate=0.01, save_name='saves/model1')

