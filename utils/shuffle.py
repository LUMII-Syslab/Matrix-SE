import tensorflow as tf


def qrol(number, q_digits, stopped_digits=0):
    """Implement cyclic left shift for quaternary numbers with stopped_positions in right side"""
    return __quaternary_shift(rol, number, stopped_digits, q_digits)


def qror(number, q_digits, stopped_digits=0):
    """Implement cyclic right shift for quaternary numbers with stopped_positions in right side"""
    return __quaternary_shift(ror, number, stopped_digits, q_digits)


def __quaternary_shift(shift_operation, number, stopped_pos, q_digits):
    """
    :param shift_operation: ror or rol function
    :param number: input number
    :param stopped_pos: How many positions leave unchanged from the right side
    :return: shifted number
    """
    bits = q_digits * 2
    stopped_bits = stopped_pos * 2

    shifted_bits = shift_operation(number >> stopped_bits, bits - stopped_bits, 2)
    unchanged_bits = number & mask(stopped_bits)
    return (shifted_bits << stopped_bits) + unchanged_bits


def quaternary_digits(number) -> int:
    bits = number.bit_length()
    bits += 1 if bits % 2 == 1 else 0

    return bits // 2


def mask(bits):
    """Generate mask of 1's for n bits"""
    return 2 ** bits - 1


def ror(x, n, p=1):
    """Bitwise rotation right"""
    return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))


def rol(x, n, p=1):
    """Bitwise rotation left"""
    return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))


def gelu(x):
    """Implements Gaussian Error Linear Unit (GELU)"""
    return x * tf.sigmoid(1.702 * x)
