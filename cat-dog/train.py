from tf_learn.utils.data_utils import NpzData, ImageData
# from inception import Model
from net import Model


logdir = 'easymodel'
n_epoch = 128

if __name__ == '__main__':
    # data = BigData('dump')
    # data = LazyImageData('data/train', lambda name: 1 if name.split('.')[0] == 'cat' else 0)
    # data = ImageData('data/train', lambda x: 1 if x.startswith("cat") else 0, handle=('resize', (300, 300)))
    data = NpzData('dump/data.npz')
    # data.set_crop((300, 300))
    model = Model(logdir)

    model.train(data, n_epoch=n_epoch, batch_size=100, validate=0.01, save_name='saves/easymodel')

