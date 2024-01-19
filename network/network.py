import numpy as np
from sklearn.datasets import load_iris , load_digits
from sklearn.model_selection import train_test_split

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def one_hot_encode(labels, num_classes):

    labels = np.array(labels , dtype=np.int32)
    encoded_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        encoded_labels[i, labels[i]] = 1
    return encoded_labels

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.uniform(size=(input_size, output_size))
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.inputs = None
        self.outputs = None
        self.error = None
        self.delta = None
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_biases = np.zeros_like(self.biases)
        
       
class NeuralNetwork:
    def __init__(self, learning_rate, batch_size , optimizer , momentum):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.layers = []
        self.momentum = momentum
    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        result = inputs
        for layer in self.layers:
            layer.inputs = result
            layer.outputs = layer.activation_function(np.dot(result, layer.weights) + layer.biases)
            result = layer.outputs
        return result

    def backward(self, labels):
        reversed_layers = self.layers[::-1]
        for i, layer in enumerate(reversed_layers):
            if i == 0:
                layer.error = 2 *(labels - layer.outputs)
                layer.delta = layer.error * layer.activation_function(layer.outputs, derivative=True)

            else:
                layer.error = reversed_layers[i-1].delta.dot(reversed_layers[i-1].weights.T)
                layer.delta = layer.error * layer.activation_function(layer.outputs, derivative=True)

        self.update_weights()

    def update_weights(self):
        for layer in self.layers:

            if self.optimizer == 'SDG':
                layer.weights +=  layer.inputs.T.dot(layer.delta) * self.learning_rate
                layer.biases +=  np.sum(layer.delta, axis=0, keepdims=True) * self.learning_rate

            elif self.optimizer == 'Momentum':
                layer.velocity_weights = self.momentum * layer.velocity_weights + layer.inputs.T.dot(layer.delta) * self.learning_rate
                layer.velocity_biases = self.momentum * layer.velocity_biases + np.sum(layer.delta, axis=0, keepdims=True) * self.learning_rate
                layer.weights += layer.velocity_weights
                layer.biases += layer.velocity_biases



    def train(self, inputs, labels, epochs):

        for epoch in range(epochs):
            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]

                prediction = self.forward(batch_inputs)
                self.backward(batch_labels)

            if (epoch) % 100 == 0:
                error = 0.5 * np.sum((prediction - batch_labels) ** 2)
                print(f"Epoch {epoch}/{epochs}, Train Loss: {error}")

    def predict(self, input_data):
        return self.forward(input_data)


def load_iris(test_size):
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    y_one_hot = one_hot_encode(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=test_size )
    return X_train , X_test , y_train , y_test 

def load_mnist(test_size):
    digits = load_digits()
    X = digits.data
    y = digits.target.reshape(-1,1)
    X = X/np.max(X)
    y_onehot = one_hot_encode(y , len(np.unique(digits.target)))
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=test_size)

    return X_train , X_test , y_train , y_test

