import tensorflow as tf
from WarpSTTensorflow import WarpST
#from AffineST import AffineST
from AffineSTTensoeflow import AffineST
from ops import *


class CNN(object):
    def __init__(self, name,batch, is_train):
        self.name = name
        self.batch = batch
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


            #input, name, dim, filter_size, strides, padding, activtion_func, batch_norm, is_train
            #Channels: 64 Features Map:256
            x1 = conv2d(x, "conv1", 64, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)
            x1 = conv2d(x1, "conv2", 64, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)

            #Channels: 128 Features Map:128
            x2 = tf.nn.max_pool2d(x1, [1,2,2,1], [1,2,2,1], "SAME")
            x2 = conv2d(x2, "conv3", 128, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)
            x2 = conv2d(x2, "conv4", 128, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)

            #Channels: 256 Features Map:64
            x3 = tf.nn.max_pool2d(x2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            x3 = conv2d(x3, "conv5", 256, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)
            x3 = conv2d(x3, "conv6", 256, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)



            #input, name, dim, filter_size, num_batch, output_shape, activtion_func, batch_norm, is_train
            #Channels: 256 Features Map:128
            x4 = deconv2d(x3, "deconv1", 256, 3, self.batch, 128, tf.nn.relu, True, self.is_train)
            x4 = tf.concat([x2, x4], -1)
            #Channels: 128 Features Map:128
            x4 = conv2d(x4, "conv7", 128, 3, 1,
                       "SAME",  tf.nn.relu, True, self.is_train)
            x4 = conv2d(x4, "conv8", 128, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)


            #Channels: 128 Features Map:256
            x5 = deconv2d(x4, "deconv2", 128, 3, self.batch, 256, tf.nn.relu, True, self.is_train)
            x5 = tf.concat([x5, x1], -1)
            #Channels: 256 Features Map:256
            x5 = conv2d(x5, "conv9", 64, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)
            x5 = conv2d(x5, "conv10", 64, 3, 1,
                       "SAME", tf.nn.relu, True, self.is_train)

            #Channels: 2 Features Map:256
            x6 = conv2d(x5, "conv11", 2, 1, 1,
                       "SAME", False, False, self.is_train)

            x7 = tf.compat.v1.layers.Flatten()(x6)

            x7 = tf.keras.layers.Dense(1024)(x7)

            x7 = tf.compat.v1.layers.Dense(64)(x7)

            initializer = tf.keras.initializers.Zeros()
            bias = tf.keras.initializers.constant([1, 0, 0, 0, 1, 0])

            x8 = tf.compat.v1.layers.Dense(6, kernel_initializer=initializer, bias_initializer=bias)(x7)


        if self.reuse is None:
            self.var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.var_list)
            self.reuse = True

        return x8

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

        self.vCNN = CNN("vector_CNN", config.batch_size, is_train=self.is_train)

        # vector map & moved image
        self.v = self.vCNN(self.xy)

        # Teste para mudar a transformação
        # Precisa mudar a rede para retornar os parâmetros certos

        #self.z = WarpST(self.x, self.v, config.im_size)
        self.z = AffineST(self.x, self.v, config.im_size)

        if self.is_train:
            self.loss = ncc(self.y, self.z)

            self.optim = tf.compat.v1.train.AdamOptimizer(config.lr)
            self.train = self.optim.minimize(
                - self.loss, var_list=self.vCNN.var_list)


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
