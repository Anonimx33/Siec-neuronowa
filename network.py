import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import data
from functools import partial


from layer import Layer

PATH_TRAIN_IMAGES = "data/train-images-idx3-ubyte"
PATH_TRAIN_LABELS = "data/train-labels-idx1-ubyte"

PATH_TEST_IMAGES = "data/t10k-images-idx3-ubyte"
PATH_TEST_LABELS = "data/t10k-labels-idx1-ubyte"

COL_SIZE = 28

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriative_sigmoid_test(x):
    return sigmoid(x) * (1-sigmoid(x))

class Network:
    def __init__(self, list_of_numbers_of_neurons, learning_rate, activation, deriative):
        self.number_of_neurons = list_of_numbers_of_neurons
        self.learning_rate = learning_rate
        self.layers = []
        self.erors = None
        for index in range(1, len(list_of_numbers_of_neurons)):
            layer = Layer(list_of_numbers_of_neurons[index], list_of_numbers_of_neurons[index-1], activation, deriative)
            self.layers.append(layer)
            

    def feed_forward(self, inputs: np.array) -> np.array:
        last_out = inputs.copy()
        for index in range(len(self.layers)):
            last_out = self.layers[index].activation(last_out)
        return last_out
    
    
    def feed_back(self, inputs: np.array, y: np.array):
        result_from_network = self.feed_forward(inputs)
        
        last_layer = self.layers[-1]
        cost_derivative = last_layer.y - y
        sigmoid_derivative = deriative_sigmoid_test(last_layer.z) #z to whynik przed sigmoidą

        errors = []
        delta = cost_derivative * sigmoid_derivative
        errors.insert(0, delta) 
        for layer_index in reversed(range(len(self.layers)-1)):
            cost_derivative = np.dot(self.layers[layer_index + 1].weights.T, delta)
            sigmoid_derivative = deriative_sigmoid_test(self.layers[layer_index].z)
            delta = cost_derivative * sigmoid_derivative
            errors.insert(0, delta)
        return errors
    
    
    def train(self, mini_batch):
        delta_biases = [np.zeros(layer.biases.shape) for layer in self.layers]
        delta_weights = [np.zeros(layer.weights.shape) for layer in self.layers]

        for sample in mini_batch: # obrazek wektor wyjść
            x, y = sample
            

            errors = self.feed_back(x, y)
            (
                delta_weights_backprop,
                delta_biases_backprop,
            ) = self.get_delta_weights_and_biases_from_errors(x, errors)

            for layer_index in range(len(self.layers)):
                delta_biases[layer_index] += delta_biases_backprop[layer_index]
                delta_weights[layer_index] += delta_weights_backprop[layer_index]

        for layer_index in range(len(self.layers)):
            delta_biases[layer_index] *= 1.0 / len(mini_batch)
            delta_biases[layer_index] *= self.learning_rate

            delta_weights[layer_index] *= 1.0 / len(mini_batch)
            delta_weights[layer_index] *= self.learning_rate

        self.update_weights_and_biases(delta_weights, delta_biases)
        

    def get_delta_weights_and_biases_from_errors(self, inputs: np.array, errors):
        to_change_biases = list()
        to_change_weights = list()

        for layer_index in range(len(self.layers) - 1, -1, -1):
            delta = errors[layer_index]
            to_change_biases.insert(0, delta.copy())

            if layer_index == 0:
                to_change_weights.insert(0, np.dot(delta, inputs.T))
            else:
                to_change_weights.insert(0, np.dot(delta, self.layers[layer_index - 1].y.T))

        return to_change_weights, to_change_biases
    

    def update_weights_and_biases(
        self, delta_weights, delta_biases):
        
        for index, layer in enumerate(self.layers):
            layer.set_biases(layer.biases - delta_biases[index])
            layer.set_weights(layer.weights - delta_weights[index])

    def cost(self, y: np.array):
        return 0.5 * (y - self.layers[-1].y) ** 2

    def final_cost(self, y: np.array):
        return sum(x for x in self.cost(y))


    def dump_to_file(self, folder_name: str = "model_w_b"):
        weights = np.array([layer.weights for layer in self.layers])
        biases = np.array([layer.biases for layer in self.layers])
        np.save(f'{folder_name}/weights.npy', weights)
        np.save(f'{folder_name}/biases.npy', biases)
    
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def dloss(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

def prepare_data(images, labels):
    network_input_size = images[0].flatten().size

    return [
        ((image.flatten() / 255).reshape((network_input_size, 1)), int(label))
        for image, label in zip(images, labels)
    ]

def make_output(number: int):
    n = np.zeros((10, 1))
    n[number][0] = 1.0
    return n

def main():
    # number of pixels in photo
    train_images, train_labels = data.load_datasets() 
    network_input_size = train_images[0].flatten().size

    # numbers in <0, 9>
    network_output_size = 10

    learning_rate = 0.1

    network = Network([network_input_size, 3, 5, network_output_size],
                      0.1, sigmoid, deriative_sigmoid_test)

    train_data = prepare_data(train_images, train_labels)
    np.random.shuffle(train_data)

    test_images, test_labels = data.load_datasets(set_name="test") 
    test_data = prepare_data(test_images, test_labels)
    np.random.shuffle(test_data)

    number_of_epochs = 10
    batch_size = 15
    test_cost = np.empty([0,1])
    train_cost = np.empty([0,1])
    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch}")
        for i in range(0, len(train_data) - batch_size, batch_size):
            mini_batch: List[Tuple[np.array, np.array]] = []

            for _, number in enumerate(train_data[i: i + batch_size]):
                pixels, number_label = number
                results: np.array = make_output(number_label)
                mini_batch.append((pixels, results))

            network.train(mini_batch)
            #network.dump_to_file()
        cost_sum = 0
        for _, number in enumerate(train_data):
            pixels, number_label = number
            output = network.feed_forward(pixels)
            result: int = int(number_label)
            results: np.array = make_output(number_label)
            cost_sum += network.final_cost(results)
        cost = cost_sum / len(train_data)
        print(cost)
        train_cost = np.append(train_cost, cost)
        for _, number in enumerate(test_data):
            pixels, number_label = number
            output = network.feed_forward(pixels)
            result: int = int(number_label)
            results: np.array = make_output(number_label)
            cost_sum += network.final_cost(results)
        cost = cost_sum / len(test_data)
        print(cost)
        test_cost = np.append(test_cost, cost)
    x = np.linspace(0,number_of_epochs, number_of_epochs)
    print(test_cost)
    plt.plot(x, train_cost,'-b', label="train data")
    plt.plot(x, test_cost,'-r', label="test data")
    plt.title("Cost for train and test data")
    plt.xlabel("Number of epochs")
    plt.ylabel("Cost")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
