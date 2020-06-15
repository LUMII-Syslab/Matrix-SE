import math

import numpy as np


def matrix_to_vector(matrix):
    """
    Converts matrix to vector according to Z-order-curve readout.
    """
    w = len(matrix)
    h = len(matrix[0])

    assert w == h, "Matrix dimensions should be equal"
    assert math.log(w * h, 4).is_integer(), "Total matrix element count should be power of 4"

    return __matrix_to_vector(matrix, w, 0, 0)


def __matrix_to_vector(matrix, length, x, y):
    if length == 1:
        return [matrix[x][y]]

    mid = length // 2

    res = []
    res += __matrix_to_vector(matrix, mid, x, y)
    res += __matrix_to_vector(matrix, mid, x, y + mid)
    res += __matrix_to_vector(matrix, mid, x + mid, y)
    res += __matrix_to_vector(matrix, mid, x + mid, y + mid)

    return res


def vector_to_matrix(vector):
    """
    Converts vector to matrix according to Z-order-curve readout.
    """
    length = len(vector)
    assert math.log(length, 4).is_integer(), "Total vector element count should be power of 4"
    return __vector_to_matrix(vector, 0, length)


def __vector_to_matrix(vector, start_pos, length):
    if length == 4:
        mid = start_pos + 2
        return [vector[start_pos:mid], vector[mid:start_pos + 4]]

    new_length = length // 4

    pos = [i for i in range(start_pos, start_pos + length, new_length)]

    first = __vector_to_matrix(vector, pos[0], new_length)
    second = __vector_to_matrix(vector, pos[1], new_length)
    third = __vector_to_matrix(vector, pos[2], new_length)
    fourth = __vector_to_matrix(vector, pos[3], new_length)

    res = []
    res += [a + b for a, b in zip(first, second)]
    res += [a + b for a, b in zip(third, fourth)]
    return res


def pad_with_zeros(tensor: np.ndarray, padded_shape: iter):
    assert len(tensor.shape) == len(padded_shape), "Shapes should be equal"

    pad_width = []
    for max_size, shape in zip(padded_shape, tensor.shape):
        if max_size == -1:
            pad_width.append((0, 0))  # No padding for this dimension
        else:
            pad_width.append((0, max_size - shape))

    return np.pad(tensor, pad_width=pad_width, mode='constant', constant_values=0)
