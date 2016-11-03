from tf_learn.utils.data_utils import BigData, LazyImageData


images = LazyImageData('data/train', lambda name: 1 if name.startswith('cat') else 0)
images.set_resize((300, 300))
meta = [
    {'shape': [None]},
    {'shape': [None, 300, 300, 3]}
]
BigData.dump(images, 'dump', meta)
