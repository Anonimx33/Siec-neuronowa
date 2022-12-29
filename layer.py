import numpy as np

class Layer:
    def __init__(self, number_of_neurons, number_of_inputs, activation_function, deriative_function):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs
        self.activation_function = activation_function
        self.deriative_function = deriative_function
        self.weights = np.random.uniform(-1,1,(self.number_of_neurons, self.number_of_inputs))
        self.biases = np.zeros((self.number_of_neurons,1))
        self.z = np.zeros((self.number_of_neurons,1))
        self.y = np.zeros((self.number_of_neurons,1))

    def activation(self, inputs):
        self.z = np.dot(self.weights, inputs) + self.biases
        self.y = self.activation_function(self.z)
        return self.y
    
    def deriative(self, inputs):
        self.z = np.dot(self.weights, inputs) + self.biases
        self.y = self.deriative_function(self.z)
        return self.y
    
    def set_biases(self, new_biases: np.array):
        self.biases = new_biases

    def set_weights(self, new_weights: np.array):
        self.weights = new_weights