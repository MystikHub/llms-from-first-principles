import math
import random
import matplotlib.pyplot as plt

# Test with y = -3.5x + 7


def true_function(x): return -3.5 * x + 7


# Values will be +/- 2 within the target function
variation_range = 20
variation_offset = -10

# Use test values between -50 and + 50
x_range = 100
x_offset = -50
domain_start = 0 + x_offset
domain_end = domain_start + x_range

# Make 10,000 and 1,00 random values each
dataset_size = 10000
dataset_x = []
dataset_y = []

learning_rate = 0.01

for x in range(dataset_size):
    # Pick a random number from the domain
    value_x = random.random() * x_range + x_offset
    dataset_x.append(value_x)

    value_y = true_function(value_x) + (random.gauss()
                                        * variation_range + variation_offset)
    dataset_y.append(value_y)

# The neuron is a function of y = mx + b
# Initialise both parameters to random values
neuron_m = random.random() * 20 - 10
neuron_b = random.random() * 20 - 10


def mean_squared_error(reference_y, predictions_y):
    sum_of_squared_errors = 0

    # Cost function is (sum(d_i - p_i) ** 2) / (dataset_size)
    for i in range(dataset_size):
        error = predictions_y[i] - reference_y[i]
        squared_error = math.pow(error, 2)
        sum_of_squared_errors += squared_error

    return sum_of_squared_errors / len(dataset_y)


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plot_dataset_points = ax.scatter(dataset_x, dataset_y, s=1)
plot_model, = ax.plot([domain_start, domain_end], [
    neuron_m * domain_start + neuron_b, neuron_m * domain_end + neuron_b], color="red")

train_iterations = 1000
for train_iteration in range(train_iterations):
    # Do one forward pass on all the values
    forward_pass_y = []
    for test_x in dataset_x:
        prediction = neuron_m * test_x + neuron_b
        forward_pass_y.append(prediction)

    forward_pass_error = mean_squared_error(dataset_y, forward_pass_y)
    print(f"forward pass error: {forward_pass_error}")

    # Do the backward pass
    # d(cost)/dm
    dcdm = 0
    for i in range(dataset_size):
        # with x = 2.5
        # truth: -1.75
        # model m = 3.5, b = 8
        # model before: 16.75
        # model m+1: 4.5, b = 8
        # model after: 19.25
        # loss before (without square): 18.5
        # loss after (without square): 21.0
        # change: 1.5
        # loss before (with square): 342.25
        # loss after (with square): 441.0
        # change: 98.75

        value = 2 * (neuron_m - dataset_y[i])
        dcdm += value
    dcdm /= dataset_size

    # d/db
    dcdb = 0
    for i in range(dataset_size):
        value = 2 * (neuron_b - dataset_y[i])
        dcdb += value
    dcdb /= dataset_size

    # update parameters
    neuron_m = learning_rate * (-1 * dcdm) + neuron_m
    neuron_b = learning_rate * (-1 * dcdb) + neuron_b

    # Show the input and model
    plot_model.set_ydata([neuron_m * domain_start + neuron_b,
                         neuron_m * domain_end + neuron_b])

    plt.xlabel('Input')  # Label for x-axis
    plt.ylabel('Output')  # Label for y-axis
    plt.title(
        f'A single neuron. Parameters: y = {neuron_m:1f}x + {neuron_b:1f}')
    # plt.show()
    # plt.draw()
    # plt.pause(0.001)
    # plt.clf()
    fig.canvas.draw()
    fig.canvas.flush_events()
