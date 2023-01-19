import os
import cv2
import numpy
import pyplot as plt
import tensorflow as tf

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

tmodel = tf.keras.models.load_model('trained.model')
loss, acc = tmodel.evaluate(x_test, y_test)

img_num = 1
while os.path.isfile(f'digits/digit{img_num}.png'):
	try:
		img = cv2.imread(f'digits/digit{img_num}.png')[:,:,0]
		img = np.invert(img)
		img = np.array([img])
		prediction = model.predict(img)
		print(f"This digit is probably a {np.argmax(prediction)}")
		plt.imshow(img[0], cmap=plt.cm.binary)
		plt.show()
	except:
		print(f"Error! Something wrong with digit{img_num}.png file!")
	finally:
		img_num += 1