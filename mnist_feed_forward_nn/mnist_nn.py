import random
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
dataset_size = ds['train'].shape[0]
grayscale_max = 256

hidden_layer_size = 8
hidden_layer_count = 8
output_size = 10 # One output for each class (handwritten digit)

image_width = 28
image_height = 28
network_input_size = image_width * image_height

class neuron:
    def __init__(self, input_count):
        self.weights = []
        self.bias = 2 * random.random() - 1
        for i in range(input_count):
            # Initialise to a random positive or negative number
            self.weights.append(2 * random.random() - 1)

    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("The size of the neuron's input does not correspond with the size used to initialise the weights")

        output = 0
        for i in range(len(self.weights)):
            output += inputs[i] * self.weights[i]

        output += self.bias
        return output

class hidden_layer:
    def __init__(self, input_count, output_count):
        self.neurons = []
        self.input_count = input_count
        for i in range(output_count):
            self.neurons.append(neuron(input_count))
        self.outputs = [0] * output_count

    def forward(self, inputs: list):
        if len(inputs) != self.input_count:
            raise ValueError("The size of the layer's input does not correspond with the size used to initialise the layer")

        for index, neuron in enumerate(self.neurons):
            self.outputs[index] = neuron.forward(inputs)

        return self.outputs

# List of neuron layers
first_layer = hidden_layer(network_input_size, hidden_layer_size)
network = [first_layer]
# Use layer_count - 2: remove one for the first and last layer
for i in range(hidden_layer_count - 2):
    network.append(hidden_layer(hidden_layer_size, hidden_layer_size))
network.append(hidden_layer(hidden_layer_size, output_size))

# Squared error
def cost_function_single(prediction, expected):
    return (prediction - expected) ** 2

def cost_function_multiple(prediction: list, expected: list):
    total = 0
    for i in range(dataset_size):
        total += cost_function_single(prediction[i], expected[i])

training_iterations = 1000
def train():
    image_index = 0
    # Do a forward pass
    # Get the network inputs from the image
    network_inputs = list(ds['train'][image_index]['image'].getdata())
    # Normalise the input values
    for index, pixel_value in enumerate(network_inputs):
        network_inputs[index] = pixel_value / grayscale_max

    network_outputs = network_inputs
    for network_layer in network:
        network_outputs = network_layer.forward(network_outputs)

    print("Forward pass complete")
    for index, value in enumerate(network_outputs):
        print(f"Output {index} value: {value}")

    # Do the backward pass

    return

for i in range(training_iterations):
    train()
