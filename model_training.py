import tensorflow as tf
from tensorflow.keras import models, layers

# #Dataset preparation
#
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000,28,28,1)).astype('float32') / 255
X_test = X_test.reshape((10000,28,28,1)).astype('float32') / 255 / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# #Model Building

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train,y_train, epochs= 5, validation_split=0.1)

model.save("model.keras")

# model = tf.keras.models.load_model('model.keras')
#
# loss, accuracy = model.evaluate(X_test,y_test)
#
# print(f"loss:{loss}")
# print(f"accuracy:{accuracy}")