from tf_learn.utils.data_utils import LazyImageData, BigData
from net import Model


logdir = 'model1'
n_epoch = 128

if __name__ == '__main__':
    data = BigData('dump')
    # data = LazyImageData('data/train', lambda name: 1 if name.split('.')[0] == 'cat' else 0)
    # data.set_crop((300, 300))
    model = Model(logdir)

    model.train(data, n_epoch=n_epoch, batch_size=64, validate=0.01, save_name='saves/model1')

