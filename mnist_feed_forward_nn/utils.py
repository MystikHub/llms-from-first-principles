import random

def make_random_list(length: int):
    new_list = []

    for i in range(length):
        new_list.append(2 * random.random() - 1)

    return new_list