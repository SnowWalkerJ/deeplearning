from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Dense, Flatten
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler, ProgbarLogger, ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
import numpy as np


train_data = np.load("dump/train.npz")
train_y, train_x = train_data['arr_0'], train_data['arr_1']
train_y = to_categorical(train_y, 2)
valid_data = np.load("dump/validate.npz")
valid_y, valid_x = valid_data['arr_0'], valid_data['arr_1']
valid_y = to_categorical(valid_y, 2)


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2, activation="softmax")(x)
model = Model(input=base_model.input, output=x)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

callbacks = [
    ProgbarLogger(),
    LearningRateScheduler(lambda x: 0.01 * 0.96 ** (x / 3.0)),
    ModelCheckpoint("saves/kmodel.{epoch:02d}-{val_loss:.2f}.hdf5"),
    # EarlyStopping(min_delta=0.001, patience=5),
    TensorBoard("logs/kmodel", 1)
]

train_gen = ImageDataGenerator(samplewise_std_normalization=True, rotation_range=20, zoom_range=0.2, horizontal_flip=False)
valid_gen = ImageDataGenerator(samplewise_std_normalization=True, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
model.fit_generator(train_gen.flow(train_x, train_y),
                    len(train_y),
                    nb_epoch=256,
                    validation_data=valid_gen.flow(valid_x, valid_y),
                    nb_val_samples=125,
                    nb_worker=4,
                    callbacks=callbacks)
