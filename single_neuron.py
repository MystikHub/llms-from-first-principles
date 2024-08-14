import random
import matplotlib.pyplot as plt

# Test with y = -3.5x + 7
true_function = lambda x: -3.5 * x + 7

# Values will be +/- 2 within the target function
variation_range = 20
variation_offset = -10

# Use test values between -500 and + 500
x_range = 100
x_offset = -50

# Make 10,000 and 1,00 random values each
test_values = 10000
dataset_x = []
dataset_y = []

for x in range(test_values):
    # Pick a random number from the domain
    value_x = random.random() * x_range + x_offset
    dataset_x.append(value_x)

    value_y = true_function(value_x) + (random.gauss() * variation_range + variation_offset)
    dataset_y.append(value_y)

plt.scatter(dataset_x, dataset_y)
plt.xlabel('Input')  # Label for x-axis
plt.ylabel('Output')  # Label for y-axis
plt.title('Plot of X vs Y')  # Title of the plot
plt.show()