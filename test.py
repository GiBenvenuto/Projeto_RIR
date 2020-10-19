import tensorflow as tf
x = tf.constant([[1, 1, 1], [1, 1, 1]])
y = tf.reduce_sum(x)
print(x)
print(y)