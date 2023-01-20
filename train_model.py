import tensorflow as tf

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

nmodel = tf.keras.models.Sequential()
nmodel.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
nmodel.add(tf.keras.layers.Dense(128, activation='relu'))
nmodel.add(tf.keras.layers.Dense(128, activation='relu'))
nmodel.add(tf.keras.layers.Dense(10, activation='softmax'))

nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

nmodel.fit(x_train, y_train, epochs=5)
nmodel.save('trained.model')