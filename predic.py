from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import cv2
from keras import backend as K
K.set_image_dim_ordering('th')



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 250, 250)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(13))
model.add(Activation('softmax'))

model.load_weights('first_try.h5')

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

im = cv2.resize(cv2.imread('/images_dataset/Test Directory/65408.jpg'),(250,250)).astype(np.float32)
ans = model.predict(im)
print ans
