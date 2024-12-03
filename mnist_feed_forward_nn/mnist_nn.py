"Trains a basic neural network to recognise handwritten digits"
import math
import random
import threading
import time
from datasets import load_dataset

import utils
from matrix import Matrix

print("Loading MNIST dataset")
ds = load_dataset("ylecun/mnist", keep_in_memory=True)
print("MNIST dataset loaded")

GRAYSCALE_MAX = 255
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

HIDDEN_LAYER_SIZE = 16
HIDDEN_LAYER_COUNT = 2
TOTAL_LAYER_COUNT = HIDDEN_LAYER_COUNT + 1

INPUT_LAYER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_LAYER_SIZE = 10  # One output for each class (handwritten digit)

TRAINING_ITERATIONS = 1000
VALIDATION_SUBSET_SIZE = 100
TRAINING_SUBSET_SIZE = 200

LEARNING_RATE = 1e-1
DROPOUT_RATIO = 1e-3

update_parameters_lock = threading.Lock()

# Each layer needs two matrices: one for weights and one for biases
layer_weights = []
layer_biases = []


def randomise_weights():
    """Sets each weight and bias to a random value"""

    for layer_index in range(TOTAL_LAYER_COUNT):
        if layer_index == 0:
            # input layer -> hidden layer
            layer_weights.append(Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE).randomise())
            layer_biases.append(Matrix(1, HIDDEN_LAYER_SIZE).randomise())
        elif layer_index == HIDDEN_LAYER_COUNT:
            # hidden layer -> output layer
            layer_weights.append(Matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE).randomise())
            layer_biases.append(Matrix(1, OUTPUT_LAYER_SIZE).randomise())
        else:
            # hidden layer -> hidden layer
            layer_weights.append(Matrix(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE).randomise())
            layer_biases.append(Matrix(1, HIDDEN_LAYER_SIZE).randomise())


def forward_pass(input_pixels_1d_list: list):
    """Infers a handwritten digit from a grayscale image"""

    normalised_inputs = [pixel_value / GRAYSCALE_MAX for pixel_value in input_pixels_1d_list]
    network_input_matrix = Matrix(1, INPUT_LAYER_SIZE)
    network_input_matrix.set_rows_columns([normalised_inputs])

    layer_output = network_input_matrix
    layer_output_cache = []  # Save the output of each layer for later use in training

    for weights, biases in zip(layer_weights, layer_biases):
        layer_output = layer_output.multiply(weights)
        layer_output = layer_output + biases
        layer_output_cache.append(layer_output)

    return layer_output_cache, utils.softmax(layer_output)


def calculate_network_loss():
    """
    Does a forward pass of the model and compares the result against the expected output over the
    validation set
    """

    total_loss = 0
    validation_set = ds["test"][:TRAINING_SUBSET_SIZE]

    for dataset_index in range(VALIDATION_SUBSET_SIZE):
        validation_image = validation_set["image"][dataset_index]

        # Shape: 1x10
        _, network_output = forward_pass(list(validation_image.getdata()))

        expected = utils.identity_distribution(validation_set["label"][dataset_index], OUTPUT_LAYER_SIZE)
        total_loss += utils.cross_entropy(network_output.data[0], expected)

    average_loss = total_loss / VALIDATION_SUBSET_SIZE
    return average_loss


def calculate_softmax_gradients(output, output_gradients):
    """
    Calculates the partial gradients for a softmax layer

    See this video for a great explanation of the mathematics behind the calculation:
    https://www.youtube.com/watch?v=AbLvJVwySEo
    """

    # "M" here refers to the same matrix from the video mentioned above. The data in this matrix
    # will be a square matrix, where every row is an array of values corresponding to the same
    # softmax output value
    matrix_m = Matrix(OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE)
    matrix_m_data = []
    for softmax_output_i in output.get_row(0):
        matrix_m_data.append([softmax_output_i] * OUTPUT_LAYER_SIZE)
    matrix_m.set_rows_columns(matrix_m_data)

    m_transpose = Matrix(OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE)
    m_transpose.set_rows_columns(matrix_m_data)
    m_transpose.transpose()

    identity_matrix = Matrix(OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE)
    identity_matrix.make_identity()

    loss_function_gradients_matrix = Matrix(1, OUTPUT_LAYER_SIZE)
    loss_function_gradients_matrix.set_rows_columns([output_gradients])
    loss_function_gradients_matrix.transpose()

    softmax_gradients = matrix_m.elementwise_multiply(identity_matrix - m_transpose)
    softmax_gradients = softmax_gradients.multiply(loss_function_gradients_matrix)
    softmax_gradients.transpose()

    return softmax_gradients


def calculate_network_gradients(input_pixels_1d_list, expected_outputs):
    """
    Does a forward pass of the network and compares the model's output to the expected output to
    calculate partial gradients for the network parameters
    """

    # First, do a forward pass to get the predicted outputs
    normalised_inputs = [pixel_value / GRAYSCALE_MAX for pixel_value in input_pixels_1d_list]
    layer_output_cache, model_output = forward_pass(normalised_inputs)
    network_input = Matrix(1, INPUT_LAYER_SIZE)
    network_input.set_rows_columns([normalised_inputs])

    # Go layer by layer and calculate partial gradients
    #
    # Step 1: Get the partial gradients of the model's output with relation to the loss function,
    # which is cross-entropy loss. For this funciton, the partial gradient for each predicted
    # value is: -log(prediction_i)
    loss_function_gradients = [-math.log(value) for value in expected_outputs]

    # Step 2: Get the partial gradients of the softmax function
    softmax_gradients = calculate_softmax_gradients(model_output, loss_function_gradients)

    # Keep track of the previous layer's partial gradients with relation to the loss function
    dLdY = softmax_gradients

    all_layer_weight_gradients = []
    all_layer_bias_gradients = []

    # The strange range function below loops from the last layer's index to 0
    for layer_index in range(TOTAL_LAYER_COUNT - 1, 0 - 1, -1):

        current_layer_weights = layer_weights[layer_index]
        current_layer_biases = layer_biases[layer_index]
        current_layer_node_count = dLdY.columns

        layer_weight_gradients = Matrix(current_layer_weights.rows, current_layer_weights.columns)
        layer_bias_gradients = Matrix(1, current_layer_biases.columns)

        if layer_index == 0:
            previous_layer_output = network_input
        else:
            previous_layer_index = layer_index - 1
            previous_layer_output = layer_output_cache[previous_layer_index]

        previous_layer_node_count = previous_layer_output.columns

        # The partial gradients for the weights for each node on this layer is equal to the previous
        # layer's outputs
        layer_weight_gradients_data = []
        for current_layer_index in range(current_layer_node_count):
            current_node_partial_gradients = previous_layer_output.get_row(0)

            # Apply the chain rule
            current_node_dl_dy = dLdY.get(0, current_layer_index)
            current_node_partial_gradients = [value * current_node_dl_dy for value in current_node_partial_gradients]

            # Columns correspond to each node in this layer. Save the partial gradient data to the
            # rows, then get the transpose of the partial gradients (for simpler understanding and
            # code)
            layer_weight_gradients_data.append(current_node_partial_gradients)

        layer_weight_gradients.set_rows_columns(layer_weight_gradients_data)
        layer_weight_gradients.transpose()
        all_layer_weight_gradients.append(layer_weight_gradients)

        # New biases are all 1. Apply the chain rule at the same time, using the previous layer's
        # derivatives
        layer_bias_gradients_data = dLdY.get_row(0)
        layer_bias_gradients.set_rows_columns([layer_bias_gradients_data])
        all_layer_bias_gradients.append(layer_bias_gradients)

        # Begin the calculation for this layer's weights' partial gradients. We have the derivatives
        # for this layer's weights with relation to to the output (dL/dY).
        #
        # Calculate and backpropagate dL/dX (dL/dY * dY/dX)
        # In other words, find out "How does changing each of the previous layer's node output
        # affect the loss function?"
        #
        # For each previous layer's node, this is:
        # sum((node * weight_previous_current + bias_current) * dLdY_current_node)
        dLdX = [0] * previous_layer_node_count
        for previous_layer_index in range(previous_layer_node_count):
            for current_layer_index in range(current_layer_node_count):
                prev_value = 1  # The partial gradient starts out at 1x

                weight = current_layer_weights.get(previous_layer_index, current_layer_index)
                bias = current_layer_biases.get(0, current_layer_index)
                current_node_contribution = prev_value * weight + bias

                # Apply the chain rule
                current_node_contribution = current_node_contribution * dLdY.get(0, current_layer_index)
                dLdX[previous_layer_index] += current_node_contribution

                # dLdY_here = dLdY.get_row(0)[output_node_index]
                # dYdX_here = current_layer_weights.get_row(input_node_index)[output_node_index]
                # dLdX_here = dLdY_here * dYdX_here
                # dLdX[input_node_index] += dLdX_here

        # Set dL/dY to this layer's dL/dX for the next layer
        dLdY.set_rows_columns([dLdX])

    # We did a backwards pass, so the partial gradients are in reverse order
    all_layer_weight_gradients.reverse()
    all_layer_bias_gradients.reverse()

    return all_layer_weight_gradients, all_layer_bias_gradients


def update_parameters(weight_gradients, bias_gradients):
    """Updates the parameters in the network to (hopefully) reduce the error"""

    with update_parameters_lock:
        # Do basic gradient descent on every layer's weights
        for layer_index in range(TOTAL_LAYER_COUNT):
            # weights + ((gradients * -1) * learning rate)
            negative_gradients = weight_gradients[layer_index].multiply_scalar(-1)
            learning_gradients = negative_gradients.multiply_scalar(LEARNING_RATE)
            layer_weights[layer_index].elementwise_add(learning_gradients)

        # Do basic gradient descent on every layer's biases
        for layer_index in range(TOTAL_LAYER_COUNT):
            # biases + ((gradients * -1) * learning rate)
            negative_gradients = bias_gradients[layer_index].multiply_scalar(-1)
            learning_gradients = negative_gradients.multiply_scalar(LEARNING_RATE)
            layer_biases[layer_index].elementwise_add(learning_gradients)


def training_iteration(dataset):
    """
    Does a forward pass, calculates parameter gradients, and updates them across the whole dataset
    """
    dataset_size = len(dataset["image"])

    # Do one training iteration by calculating the partial gradients of every
    # parameter in the network (averaged over every image)
    accumulate_weight_gradients = []
    accumulate_bias_gradients = []
    for dataset_index in range(dataset_size):
        network_input_data = list(dataset["image"][dataset_index].getdata())
        expected_outputs = utils.identity_distribution(dataset["label"][dataset_index], OUTPUT_LAYER_SIZE)

        weight_gradients, bias_gradients = calculate_network_gradients(network_input_data, expected_outputs)
        if len(accumulate_weight_gradients) == 0:
            # First accumulation
            accumulate_weight_gradients = weight_gradients
            accumulate_bias_gradients = bias_gradients
        else:
            # Accumulate the weights over the whole dataset
            for i, layer_weight_gradients in enumerate(weight_gradients):
                accumulate_weight_gradients[i].elementwise_add(layer_weight_gradients)

            # Same for the biases
            for i, layer_bias_gradients in enumerate(bias_gradients):
                accumulate_bias_gradients[i].elementwise_add(layer_bias_gradients)

    for i in range(TOTAL_LAYER_COUNT):
        accumulate_weight_gradients[i] = accumulate_weight_gradients[i].divide_scalar(dataset_size)
        accumulate_bias_gradients[i] = accumulate_bias_gradients[i].divide_scalar(dataset_size)

    update_parameters(accumulate_weight_gradients, accumulate_bias_gradients)


def dropout_parameters():
    """Randomly re-assigns a random value to a small percentage of parameters"""
    # Iterate through every weight
    for layer_index, layer in enumerate(layer_weights):
        for row_index in range(layer.rows):
            for column_index in range(layer.columns):

                # Randomly select some weights
                select_this_weight = random.random() <= DROPOUT_RATIO

                # Replace the weight with a random value
                if select_this_weight:
                    layer_weights[layer_index].data[row_index][column_index] = random.random() - 1

    # Same for the biases
    for layer_index, layer in enumerate(layer_biases):
        for row_index in range(layer.rows):
            for column_index in range(layer.columns):

                select_this_bias = random.random() <= DROPOUT_RATIO

                if select_this_bias:
                    layer_biases[layer_index].data[row_index][column_index] = random.random() - 1


if __name__ == "__main__":
    randomise_weights()

    for i in range(TRAINING_ITERATIONS):
        print("Calculating network loss")
        network_loss = calculate_network_loss()
        print(f"Average cross-entropy loss: {network_loss:.5f}")

        training_start = time.time()
        training_iteration(ds["train"][:TRAINING_SUBSET_SIZE])
        dropout_parameters()
        training_duration = time.time() - training_start

        print(f"One Training step over {TRAINING_SUBSET_SIZE} images took {training_duration:.2f}s")
