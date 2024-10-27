import math
import utils
from datasets import load_dataset
from matrix import Matrix
from vector import Vector

print("Loading MNIST dataset")
ds = load_dataset("ylecun/mnist")
print("MNIST dataset loaded")
grayscale_max = 255

hidden_layer_size = 3
hidden_layer_count = 3
total_layer_count = hidden_layer_size + 1

image_width = 28
image_height = 28
network_input_size = image_width * image_height
network_output_size = 10 # One output for each class (handwritten digit)

# Each layer is made of two matrices: one for weights and one for biases
layer_weights = []
layer_biases = []
for layer_number in range(total_layer_count):
    if layer_number == 0:
        # input layer -> hidden layer
        layer_weights.append(Matrix(network_input_size, hidden_layer_size))
        layer_biases.append(Matrix(1, hidden_layer_size).randomise())
    elif layer_number == hidden_layer_size:
        # hidden layer -> output layer
        layer_weights.append(Matrix(hidden_layer_size, network_output_size))
        layer_biases.append(Matrix(1, network_output_size).randomise())
    else:
        # hidden layer -> hidden layer
        layer_weights.append(Matrix(hidden_layer_size, hidden_layer_size))
        layer_biases.append(Matrix(1, hidden_layer_size).randomise())

# Cross entropy loss
def cost_function(prediction: list, expected: list):
    # Both inputs must have the same number of classes
    if len(prediction) != len(expected):
        raise ValueError(f"The classes in the prediction ({len(prediction)}) does not match with the expected values ({len(expected)})!")

    # Cross entropy loss between two distributions of discrete random variables
    # Based on the information from https://en.wikipedia.org/wiki/Cross-entropy
    # -∑(prediction * log(train))
    total = 0
    for class_index in range(len(prediction)):
        total += prediction[class_index] * math.log(expected[class_index])
    return total * -1

# Creates a list for the prediction class distribution
def get_expected_distribution(correct_class):
    # Cross entropy loss uses the log function, so we can't use zero
    prediction = [1e-3] * 10
    prediction[correct_class] = 1
    return prediction

def train():
    total_loss = 0
    dataset_subset_size = min(1000, ds['train'].num_rows)

    for dataset_image_index in range(dataset_subset_size):
        dataset_image = ds['train'][dataset_image_index]

        # Do a forward pass
        # Get the network inputs from the image
        network_input_data = list(dataset_image['image'].getdata())

        # Normalise the input values to the range 0 -> 1
        for index, pixel_value in enumerate(network_input_data):
            network_input_data[index] = pixel_value / grayscale_max
        
        network_input = Matrix(1, network_input_size)
        network_input.set_data([network_input_data])

        layer_output = network_input
        for (weights, biases) in zip(layer_weights, layer_biases):
            layer_output = layer_output.multiply(weights)
            layer_output = layer_output + biases

        prediction = get_expected_distribution(dataset_image['label'])
        total_loss += cost_function(layer_output.data[0], prediction)

    average_loss = total_loss / dataset_subset_size
    print(f"Average cross-entropy loss = {average_loss:.3f}")

    return

training_iterations = 1000
for i in range(training_iterations):
    train()
