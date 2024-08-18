import math
import random
import matplotlib.pyplot as plt

# Test with y = -3.5x + 7
truth_m = 5 * random.random() - 2.5
truth_b = 20 * random.random() - 10
# truth_m = -5
# truth_b = -10


def true_function(x):
    return truth_m * x + truth_b
    # return -3.5 * x + 7


# Values will be +/- 2 within the target function
variation_range = 20
# Use test values between -50 and + 50
x_range = 100
x_offset = -50
domain_start = 0 + x_offset
domain_end = domain_start + x_range

# Make 10,000 and 1,00 random values each
dataset_size = 1000
dataset_x = []
dataset_y = []

learning_rate_m = 0.0005
learning_rate_b = 0.1
error_plot = []

for x in range(dataset_size):
    # Pick a random number from the domain
    value_x = random.random() * x_range + x_offset
    dataset_x.append(value_x)

    value_y = true_function(value_x) + random.gauss() * variation_range
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

train_iterations = 50
for train_iteration in range(train_iterations):
    # Do one forward pass on all the values
    forward_pass_y = []
    for test_x in dataset_x:
        prediction = neuron_m * test_x + neuron_b
        forward_pass_y.append(prediction)

    forward_pass_error = mean_squared_error(dataset_y, forward_pass_y)
    error_plot.append(forward_pass_error)

    # Do the backward pass
    # d(cost)/dm
    dldm = 0
    dldm_momentum = 0
    dldb = 0
    dldb_momentum = 0
    for i in range(dataset_size):
        prediction = forward_pass_y[i]
        truth = dataset_y[i]
        x = dataset_x[i]

        m = neuron_m
        b = neuron_b

        # loss = ((m * x + b) - truth) ** 2

        # Slope of m wrt L is
        # b should not change
        # x should not change
        # We're only looking to change m
        # i.e. How does m change the loss when x and b are the same?
        #
        # find dl/dm for l(g(m)^2)
        # dl/dm = dl/dg * dg/dm
        # g(m) = (mx + b - truth)
        # dg/dm = x
        # dl/dg = 2 * g(m)
        # Since l(pred, truth) is quadratic, dl/dm will be a linear function
        # dl/dm = 2 * (mx + b - truth) * x
        # For training, the slope of dl/dm needs to be sampled
        dldm_func = lambda m: 2 * (m * x + b - truth) * x
        dldm_here = dldm_func(neuron_m)

        # dldm += dldm_here
        dldm += dldm_here

        dldb_func = lambda b: 2 * (m * x + b - truth) * 1
        dldb_here = dldb_func(neuron_b)

        dldb += dldb_here

    dldm = (dldm / dataset_size) + dldm_momentum
    dldm_momentum = dldm
    dldb = (dldb/dataset_size) + dldb_momentum
    dldb_momentum = dldb

    print(f"forward pass {train_iteration:3} error: {forward_pass_error:.3f}, dldm: {dldm:.2f}, dldm_momentum: {dldm_momentum:.2f}, dldb: {dldb:.2f}, dldb_momentum: {dldb_momentum:.2f}")

    # update parameters
    neuron_m = learning_rate_m * (-1 * dldm) + neuron_m
    neuron_b = learning_rate_b * (-1 * dldb) + neuron_b

    # Show the input and model
    plot_model.set_ydata([neuron_m * domain_start + neuron_b,
                         neuron_m * domain_end + neuron_b])

    plt.xlabel('Input')  # Label for x-axis
    plt.ylabel('Output')  # Label for y-axis
    plt.title(
        f'Dataset: {truth_m:.3f}x + {truth_b:.3f}\nModel: y = {neuron_m:.3f}x + {neuron_b:.3f}')
    # plt.show()
    # plt.draw()
    # plt.pause(0.001)
    # plt.clf()
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.clf()
plt.title("Error over training")
plt.plot(error_plot)
plt.show()