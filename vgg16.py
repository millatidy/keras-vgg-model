from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as K

# dimensions of our images
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

class VGG16:

	def __init__(self, weights_path=None, include_top=True):
		self.model = Sequential()

		self.model.add(Conv2D(64, (3,3), input_shape=input_shape, activation='relu'))
		self.model.add(Conv2D(64, (3,3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(128, (3,3), activation='relu'))
		self.model.add(Conv2D(128, (3,3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(256, (3,3), activation='relu'))
		self.model.add(Conv2D(256, (3,3), activation='relu'))
		self.model.add(Conv2D(256, (3,3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(Conv2D(512, (3,3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		if include_top:
			self.model.add(Flatten())
			self.model.add(Dense(4096, activation='relu'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(4096, activation='relu'))
			self.model.add(Dropout(0.5))
			self.model.add(Dense(1000, activation='softmax'))

		if weights_path:
			self.model.load_weights(weights_path)

	def summary(self):
		 return self.model.summary()

model = VGG16(include_top=False)
print (model.summary())