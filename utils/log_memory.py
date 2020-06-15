import tensorflow as tf

__log_cache = {}
__showed = False


def log_in_tb(obj: tf.keras.layers.Layer, name: str):
    """
    :param name: name of loggable layer
    :param obj: Callable that returns tf.Tensor object
    """
    global __showed
    global __log_cache

    if __showed:
        __showed = False
        __log_cache = {}

    def wrapper(*args, **kwargs):
        res = obj(*args, **kwargs)  # type: tf.Tensor
        length = res.shape.as_list()[1]

        if name not in __log_cache:
            __log_cache[name] = {}

        if length not in __log_cache[name]:
            __log_cache[name][length] = []

        __log_cache[name][length].append(res)

        return res

    return wrapper


def show_layer_outputs():
    global __showed
    __showed = True

    for name, l_dic in __log_cache.items():
        length, values = max(l_dic.items())
        values = tf.stack(values)
        values = values[:, 0:1, :64, :]
        l_name = name + "_" + str(length)
        tf.compat.v1.summary.image(l_name, tf.transpose(values, [3, 0, 2, 1]), max_outputs=10, family="layer_outputs")
