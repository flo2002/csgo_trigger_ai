# import all the necassary packages
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

# set image size and instance name
IMG_SIZE = 100
NAME = "csgo"

# initialize the data loader
datagen = ImageDataGenerator(
    validation_split=0.1)
# load the training data
data = datagen.flow_from_directory(
    'dataset',
    class_mode='categorical',
    batch_size=128,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset='training')
# load the validation data
valid = datagen.flow_from_directory(
    'dataset',
    class_mode='categorical',
    batch_size=128,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset='validation')

# initialize a sequential model
model = Sequential()

# create the neural network 
model.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

# initialize the monitoring tool TensorBoard
tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

# compile the neural network
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# print out the structure of the neural network
model.summary()

# start the training
model.fit(data, batch_size=256, validation_data=valid, epochs=100, shuffle=True, callbacks=[tensorboard])
# save the model
model.save(NAME+'.model')
