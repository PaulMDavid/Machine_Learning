import keras
import os.path
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# Path to saved model weights(as hdf5)
resume_weights = "fashion_mnist-cnn-best.hdf5"

batch_size=128
epochs=2
num_classes=10 

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	# 1 x 28 x 28 [number_of_channels (colors) x height x weight]
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	# 28 x 28 x 1 [height x weight x number_of_channels (colors)]
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

if os.path.isfile(resume_weights):
	print ("Resumed model's weights from {}".format(resume_weights))
	# load weights
	model.load_weights(resume_weights)

# CEE, Adam
model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adam(),
			metrics=['accuracy'])

# Checkpoint In the folder
filepath = "mnist-cnn-best.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')

# Train
model.fit(x_train, y_train,	batch_size=batch_size,
				epochs=epochs,
				verbose=1,
				validation_data=(x_test, y_test),
				callbacks=[checkpoint])

# Eval
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


