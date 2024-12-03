"""Provides various utility functions for training a neural network"""

import math
import random
from matrix import Matrix


def random_list(length: int):
    "Creates a list of `length` random items"
    new_list = []

    for i in range(length):
        new_list.append(random.random() - 0.5)

    return new_list


def cross_entropy(distribution_a: list, distribution_b: list):
    """Calculates the cross-entropy between two distributions"""
    # Both inputs must have the same number of classes
    if len(distribution_a) != len(distribution_b):
        raise ValueError(
            f"The number of elements in the first distribution ({len(distribution_a)}) is not equal"
            f" to the number of elements in the second_distribution ({len(distribution_b)})"
        )

    # Cross entropy loss between two distributions of discrete random variables
    # Based on information from https://en.wikipedia.org/wiki/Cross-entropy
    # -∑(prediction * log(train))
    total = 0
    for index, a_value in enumerate(distribution_a):
        total += a_value * math.log(distribution_b[index])

    return total * -1


def identity_distribution(one_index, length):
    "Creates a list with the given length where every value is almost 0 and one value is set to 1"

    if length <= 0:
        raise ValueError("length must be a positive integer.")

    if one_index >= length:
        raise ValueError("index must be less than the length.")

    # Cross entropy loss (used together with this function) depends on log function, so we can't
    # use zero
    small_value = 1e-3
    prediction = [small_value] * length
    prediction[one_index] = 1 - (small_value * (length - 1))

    return prediction


def softmax(distribution: Matrix):
    """Applies the softmax function to a one-row matrix, returning the result in another matrix"""

    if distribution.rows != 1:
        raise ValueError("The input to the softmax function must be a 1-row matrix")

    input_max = max(distribution.get_row(0))
    exponent_values = [math.exp(value - input_max) for value in distribution.get_row(0)]

    total = sum(exponent_values)
    normalised_values = [value / total for value in exponent_values]

    # Return the new values as another matrix
    layer_output = Matrix(1, distribution.columns)
    layer_output.set_rows_columns([normalised_values])

    return layer_output
