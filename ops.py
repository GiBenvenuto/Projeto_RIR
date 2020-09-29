import tensorflow as tf
import os
import skimage.io
import csv
import numpy as np


def conv2d(x, name, dim, k, s, p, bn, af, is_train):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', [k, k, x.get_shape()[-1], dim],
                                      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

        if bn:
            x = batch_norm(x, "bn", is_train=is_train)
        else:
            b = tf.compat.v1.get_variable('biases', [dim],
                                          initializer=tf.constant_initializer(0.))
            x += b

        if af:
            x = af(x)

    return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
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


def salve_training_data(loss, name):
    csv.field_size_limit(393216)
    with open("Files/" + name, 'w') as csvfile:
        for i in loss:
            row = str(i) + "\n"
            csvfile.write(row)

    csvfile.close()
