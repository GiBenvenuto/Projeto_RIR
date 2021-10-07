import tensorflow as tf
from WarpSTTensorflow import WarpST
from ops import *


class CNN(object):
    def __init__(self, name, batch, img_size, is_train):
        self.name = name
        self.batch = batch
        self.img_size = img_size
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):


            #Parametros:
            #input, name, dim, filter_size, strides, padding, activtion_func, batch_norm, bias, is_train
            #Channels: 64 Features Map:512
            x1 = conv2d(x, "conv1", 64, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)
            x1 = conv2d(x1, "conv2", 64, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)

            #Channels: 128 Features Map:256
            x2 = tf.nn.max_pool2d(x1, [1,2,2,1], [1,2,2,1], "SAME")
            x2 = conv2d(x2, "conv3", 128, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)
            x2 = conv2d(x2, "conv4", 128, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)

            #Channels: 256 Features Map:128
            x3 = tf.nn.max_pool2d(x2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            x3 = conv2d(x3, "conv5", 256, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)
            x3 = conv2d(x3, "conv6", 256, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)



            #Parametros:
            #input, name, dim, filter_size, num_batch, output_shape, activtion_func, batch_norm, is_train
            #Channels: 256 Features Map:256
            x4 = deconv2d(x3, "deconv1", 256, 3, self.batch, int(self.img_size/2), tf.nn.relu, True, self.is_train)
            x4 = tf.concat([x2, x4], -1)
            #Channels: 128 Features Map:256
            x4 = conv2d(x4, "conv7", 128, 3, 1,
                       "SAME",  tf.nn.relu, True, True, self.is_train)
            x4 = conv2d(x4, "conv8", 128, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)


            #Channels: 128 Features Map:512
            x5 = deconv2d(x4, "deconv2", 128, 3, self.batch, int(self.img_size), tf.nn.relu, True, self.is_train)
            x5 = tf.concat([x5, x1], -1)
            #Channels: 64 Features Map:512
            x5 = conv2d(x5, "conv9", 64, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)
            x5 = conv2d(x5, "conv10", 64, 3, 1,
                       "SAME", tf.nn.relu, True, True, self.is_train)

            #Channels: 2 Features Map:512
            x6 = conv2d(x5, "conv11", 2, 1, 1,
                       "SAME", False, False, False, self.is_train)


        if self.reuse is None:
            self.var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.compat.v1.train.Saver(self.var_list)
            self.reuse = True

        return x6

    def save(self, sess, ckpt_path):
        self.saver.save(sess, ckpt_path)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class STN(object):
    def __init__(self, sess, config, name, is_train):
        tf.compat.v1.disable_eager_execution()

        self.sess = sess
        self.name = name
        self.is_train = is_train

        # moving / fixed images
        im_shape = [config.batch_size] + config.im_size + [1]
        self.x = tf.compat.v1.placeholder(tf.float32, im_shape)
        self.y = tf.compat.v1.placeholder(tf.float32, im_shape)
        #self.tx = tf.compat.v1.placeholder(tf.float32, im_shape)
        #self.ty = tf.compat.v1.placeholder(tf.float32, im_shape)
        #Concatena as imagens de entrada
        self.xy = tf.concat([self.x, self.y], 3)

        #Chama a CNN
        self.vCNN = CNN("vector_CNN", config.batch_size, config.im_size[0], is_train=self.is_train)

        # vector map & moved image
        self.v = self.vCNN(self.xy)

        self.var = self.vCNN.var_list

        #Passa o grid encontrado para a STN
        self.z, self.tg = WarpST(self.x, self.v, config.im_size)



        if self.is_train:
            self.loss = ncc(self.y, self.z)

            self.optim = tf.compat.v1.train.AdamOptimizer(config.lr)
            self.train = self.optim.minimize(
                - self.loss, var_list=self.vCNN.var_list)



        self.sess.run(
            tf.compat.v1.global_variables_initializer())

    def fit2steps(self, batch_x, batch_y, batch_tf, batch_tm):
        _, loss = \
            self.sess.run([self.train, self.loss],
                          {self.x: batch_x, self.y: batch_y, self.tf: batch_tf, self.tm: batch_tm})
        return loss

    def fit(self, batch_x, batch_y):
        _, loss = \
            self.sess.run([self.train, self.loss],
                          {self.x: batch_x, self.y: batch_y})
        return loss

    def deploy(self, dir_path, x, y, j):
        z = self.sess.run(self.z, {self.x: x, self.y: y})

        for i in range(z.shape[0]):
            save_image_with_scale(dir_path + "/{:02d}_x.tif".format(j + 1), x[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_y.tif".format(j + 1), y[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_z.tif".format(j + 1), z[i, :, :, 0])

    def deploy2steps(self, dir_path, x, y, tx, ty, j):
        z = self.sess.run(self.z, {self.x: x, self.y: y, self.tx: tx, self.ty: ty})

        for i in range(z.shape[0]):
            #save_image_with_scale(dir_path + "/{:02d}_x.tif".format(j + 1), x[i, :, :, 0])
            #save_image_with_scale(dir_path + "/{:02d}_y.tif".format(j + 1), y[i, :, :, 0])
            save_image_with_scale(dir_path + "/{:02d}_z.tif".format(j + 1), z[i, :, :, 0])


    def grid(self, x, y):
        v = self.sess.run(self.tg, {self.x: x, self.y: y})

        return v


    def save(self, dir_path):
        self.vCNN.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.vCNN.restore(self.sess, dir_path + "/model.ckpt")

    def varlist(self, x, y):
        z = self.vCNN.var_list
        z = self.sess.run(self.var, {self.x: x, self.y: y})
        return z

