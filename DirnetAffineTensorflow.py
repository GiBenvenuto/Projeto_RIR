import tensorflow as tf
from AffineSTTensoeflow import AffineST
from ops import *


class CNN(object):
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            '''Conv2D
             x - input tensor
             name - layer name
             dim - kernel dim, 
             k - shape filter, 
             s - strides, 
             p - padding, 
             bn - bach normalization, 
             af - elu, 
             is_train - for reuse'''

            x = conv2d(x, "conv1", 64, 3, 2,
                       "SAME", True, tf.nn.relu, self.is_train)
            # x = tf.nn.max_pool2d(x, [1,2,2,1], [1,2,2,1], "SAME")

            x = conv2d(x, "conv2", 128, 3, 2,
                       "SAME", True, tf.nn.relu, self.is_train)
            # x = tf.nn.max_pool2d(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

            x = conv2d(x, "conv3", 128, 3, 2,
                       "SAME", True, tf.nn.relu, self.is_train)
            # x = tf.nn.max_pool2d(x, [1,2,2,1], [1,2,2,1], "SAME")

            x = conv2d(x, "conv4", 2, 3, 2,
                       "SAME", True, tf.nn.relu, self.is_train)

            x = tf.compat.v1.layers.Flatten()(x)

            x = tf.keras.layers.Dense(1024)(x)

            x = tf.compat.v1.layers.Dense(64)(x)

            initializer = tf.keras.initializers.Zeros()
            bias = tf.keras.initializers.constant([1, 0, 0, 0, 1, 0])

            x = tf.compat.v1.layers.Dense(6, kernel_initializer=initializer, bias_initializer=bias)(x)

        if self.reuse is None:
            self.var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.var_list)
            self.reuse = True

        return x

    def save(self, sess, ckpt_path):
        self.saver.save(sess, ckpt_path)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class DIRNet(object):
    def __init__(self, sess, config, name, is_train):
        tf.compat.v1.disable_eager_execution()

        self.sess = sess
        self.name = name
        self.is_train = is_train

        # moving / fixed images
        im_shape = [config.batch_size] + config.im_size + [1]
        self.x = tf.compat.v1.placeholder(tf.float32, im_shape)
        self.y = tf.compat.v1.placeholder(tf.float32, im_shape)
        self.xy = tf.concat([self.x, self.y], 3)

        self.vCNN = CNN("vector_CNN", is_train=self.is_train)

        # vector map & moved image
        self.v = self.vCNN(self.xy)

        # Teste para mudar a transformação
        # Precisa mudar a rede para retornar os parâmetros certos
        # self.z = AffineST(self.x, self.v)
        self.z = AffineST(self.x, self.v, config.im_size)

        if self.is_train:
            self.loss = ncc(self.y, self.z)

            self.optim = tf.compat.v1.train.AdamOptimizer(config.lr)
            self.train = self.optim.minimize(
                - self.loss, var_list=self.vCNN.var_list)

        # self.sess.run(
        #  tf.variables_initializer(self.vCNN.var_list))
        self.sess.run(
            tf.compat.v1.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss = \
            self.sess.run([self.train, self.loss],
                          {self.x: batch_x, self.y: batch_y})
        return loss

    def deploy(self, dir_path, x, y):
        z = self.sess.run(self.z, {self.x: x, self.y: y})
        for i in range(z.shape[0]):
            save_image_with_scale(dir_path + "/{:02d}_x.tif".format(i + 1), x[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_y.tif".format(i + 1), y[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_z.tif".format(i + 1), z[i, :, :, 0])

    def save(self, dir_path):
        self.vCNN.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.vCNN.restore(self.sess, dir_path + "/model.ckpt")
