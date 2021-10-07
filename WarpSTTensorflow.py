import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from bicubic import bicubic_interp_2d

def WarpST(U, V, out_size, name='DeformableTransformer', **kwargs):
    """Deformable Transformer Layer with bicubic interpolation
    U : tf.float, [num_batch, height, width, num_channels].
        Input tensor to warp
    V : tf.float, [num_batch, height, width, 2]
        Warp map. It is interpolated to out_size.
    out_size: a tuple of two ints
        The size of the output of the network (height, width)
    ----------
    References :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    def _repeat(x, n_repeats):
        with tf.compat.v1.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(img)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))
        b = tf.reshape(b, [-1])

        indices = tf.stack([b, y, x], 1)

        return tf.gather_nd(img, indices)

    def _interpolate(im, x, y, out_size):
        with tf.compat.v1.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0] #Para teste = 1
            height = tf.shape(im)[1] #512
            width = tf.shape(im)[2] #512
            channels = tf.shape(im)[3] #1

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32') #512
            width_f = tf.cast(width, 'float32') #512
            out_height = out_size[0] #512
            out_width = out_size[1] #512
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32') #511
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32') #511

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            # arruma os limites dos valores entre 0 e 511
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            #get pixel value at corner coords - usa os indices para pegar os valores
            Ia = get_pixel_value(im, x0, y0)
            Ib = get_pixel_value(im, x0, y1)
            Ic = get_pixel_value(im, x1, y0)
            Id = get_pixel_value(im, x1, y1)

            # and finally calculate interpolated values.
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.compat.v1.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            grid = tf.concat([x_t_flat, y_t_flat], 0)
            return grid

    def _transform(V, U, out_size):
        with tf.compat.v1.variable_scope('_transform'):
            num_batch = tf.shape(U)[0] #Para teste = 1
            height = tf.shape(U)[1] #512
            width = tf.shape(U)[2] #512
            num_channels = tf.shape(U)[3] #1

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0] #512
            out_width = out_size[1] #512
            #Cria uma grade regular com as coordenadas das matrizes em vetores de [-1, 1]
            grid = _meshgrid(out_height, out_width)     # [2, h*w]
            grid = tf.reshape(grid, [-1])               # [2*h*w]
            grid = tf.tile(grid, tf.stack([num_batch]))           # [n*2*h*w]
            grid = tf.reshape(grid, tf.stack([num_batch, 2, -1])) # [n, 2, h*w]

            # Set source position (x+vx, y+vy)^T
            V = bicubic_interp_2d(V, out_size)
            V = tf.transpose(V, [0, 3, 1, 2])           # [n, 2, h, w]
            V = tf.reshape(V, [num_batch, 2, -1])       # [n, 2, h*w]
            #Aplica os valores encontrados pela rede a grade criada
            T_g = tf.add(V, grid)                       # [n, 2, h*w]


            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            #Interpola os valores vizinhos
            input_transformed = _interpolate(
                U, x_s_flat, y_s_flat, out_size)

            output = tf.reshape(
                input_transformed,
                tf.stack([num_batch, out_height, out_width, num_channels]))
            return output, tf.reshape(T_g, [num_batch, 2, out_width, out_height])

    with tf.compat.v1.variable_scope(name):
        output, tg = _transform(V, U, out_size)
        return output, tg

#Parte que estou mudando :) Só ignora
'''
            dim2 = width #512
            dim1 = width*height #262144
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)
'''