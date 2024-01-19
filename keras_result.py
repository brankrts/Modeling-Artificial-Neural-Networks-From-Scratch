import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import random

def load_mnist(test_size):
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = X / np.max(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def test_on_samples(model, x_test, y_test, test_count):
    for _ in range(test_count):
        sample_index = random.randint(0, len(x_test) - 1)
        sample_data = x_test[sample_index, :].reshape(1, -1)
        sample_label = y_test[sample_index]
        prediction = np.argmax(model.predict(sample_data))
        print(f"True Class: {sample_label}, Predicted Class: {prediction}")

X_train, X_test, y_train, y_test = load_mnist(test_size=0.2)

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

model = Sequential()
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))

sgd = SGD(learning_rate=0.1)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

batch_size = 32
history = model.fit(X_train, y_train_one_hot, epochs=300, batch_size=batch_size, validation_data=(X_test, y_test_one_hot))

train_loss = history.history['loss']
test_loss = history.history['val_loss']

train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

print("Train Loss:", np.min(train_loss))
print("Test Accuracy:", np.max(test_accuracy))
test_on_samples(model, X_test, y_test, 20)
