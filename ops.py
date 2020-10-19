import tensorflow as tf
import os
import skimage.io
import numpy as np

def conv2d(input, name, dim, filter_size, strides, padding, activtion_func, batch_norm, is_train):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', [filter_size, filter_size, input.get_shape()[-1], dim],
                                      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01))
        b = tf.compat.v1.get_variable('biases', [dim],
                                      initializer=tf.constant_initializer(0.))

        layer = tf.nn.conv2d(input, w, [1, strides, strides, 1], padding)

        layer += b

        if activtion_func:
            layer = activtion_func(layer)

        if batch_norm:
            layer = batch_normalization(layer, "bn", is_train=is_train)

    return layer

def deconv2d (input, name, dim, filter_size, num_batch, output_shape, activtion_func, batch_norm, is_train):

    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', [filter_size, filter_size, input.get_shape()[-1], dim],
                                      initializer=tf.keras.initializers.glorot_uniform)
        b = tf.compat.v1.get_variable('biases', [dim],
                                      initializer=tf.constant_initializer(0.))

        layer = tf.nn.conv2d_transpose(input=input, filters=w,
                                       output_shape=[num_batch, output_shape, output_shape, dim],
                                       strides=[1, 2, 2, 1], padding='SAME')
        layer += b

        if activtion_func:
            layer = activtion_func(layer)

        if batch_norm:
            layer = batch_normalization(layer, "bn", is_train=is_train)

    return layer

def batch_normalization(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.keras.layers.BatchNormalization(momentum=momentum,
                                              epsilon=epsilon,
                                              scale=True,
                                              trainable=is_train,
                                              name=name)(x)


# normalized cross correlation loss
def ncc(x, y):
    mean_x = tf.math.reduce_mean(x, [1, 2, 3], keepdims=True)
    mean_y = tf.math.reduce_mean(y, [1, 2, 3], keepdims=True)

    mean_x2 = tf.math.reduce_mean(tf.square(x), [1, 2, 3], keepdims=True)
    mean_y2 = tf.math.reduce_mean(tf.square(y), [1, 2, 3], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(
        mean_x2 - tf.square(mean_x)), [1, 2, 3], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(
        mean_y2 - tf.square(mean_y)), [1, 2, 3], keepdims=True)
    div = tf.math.divide((x - mean_x) * (y - mean_y), (stddev_x * stddev_y))
    out = tf.math.reduce_mean(div)
    return out


# gain coefficient
def gc(y, x, z):
    after = tf.math.equal(y, z)
    after = tf.math.reduce_sum(tf.cast(after, tf.float32))
    before = tf.math.equal(y, x)
    before = tf.math.reduce_sum(tf.cast(before, tf.float32))

    return tf.math.divide(after, before)


# Mean Square Error
def mse(x, y):
    return tf.math.reduce_mean(tf.square(x - y))


def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except:
        pass


def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    skimage.io.imsave(path, arr)



