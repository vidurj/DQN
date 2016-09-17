import numpy as np
import tensorflow as tf
import cv2


class DQN:
    def __init__(self, params, name):
        self.params = params
        self.network_name = name
        sel.x = tf.placeholder('float32', [None, 84, 84, 4], name=network_name + '_x')
        self.q_t = tf.placeholder('float32', [None], name=network_name + '_q_t')
        self.actions = tf.placeholder("float32", [None, params['num_act']], name=network_name + '_actions')
        self.rewards = tf.placeholder("float32", [None], name=network_name + '_rewards')
        self.terminals = tf.placeholder("float32", [None], name=network_name + '_terminals')

        # conv1
        with tf.variable_scope("CONV"):
            # conv1
            size = 8
            channels = 4
            filters = 16
            stride = 4
            w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
            b1 = tf.Variable(tf.constant(0.1, shape=[filters]))
            c1 = tf.nn.conv2d(x, w1, strides=[1, stride, stride, 1], padding='VALID')
            o1 = tf.nn.relu(tf.add(c1, b1))

            # conv2
            size = 4
            channels = 16
            filters = 32
            stride = 2
            w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
            b2 = tf.Variable(tf.constant(0.1, shape=[filters]))
            c2 = tf.nn.conv2d(o1, w2, strides=[1, stride, stride, 1], padding='VALID')
            o2 = tf.nn.relu(tf.add(c2, b2))

            # flat
            o2_shape = o2.get_shape().as_list()

        with tf.variable_scope("FC"):
            # fc3
            hiddens = 512
            dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
            o2_flat = tf.reshape(o2, [-1, dim])
            w3 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01))
            b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]))
            self.mask = tf.placeholder(tf.float32, (None, 512))
            o3_dropout = tf.nn.relu(tf.add(tf.matmul(o2_flat, w3), b3)) * self.mask

            # fc4
            hiddens = params['num_act']
            dim = 512
            w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01))
            b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]))
            y = tf.nn.softplus(tf.add(tf.matmul(o3_dropout, w4), b4))

        # Q,Cost,Optimizer
        discount = tf.constant(params['discount'])
        yj = tf.add(rewards, tf.mul(1.0 - terminals, tf.mul(discount, q_t)))
        Qxa = tf.mul(y, actions)
        Q_pred = tf.reduce_sum(Qxa, reduction_indices=1)
        diff = tf.sub(yj, Q_pred)

        diff_square = tf.mul(tf.constant(0.5), tf.pow(diff, 2))
        cost = tf.reduce_sum(diff_square)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(params['lr'], params['rms_decay'], 0.0,
                                              params['rms_eps'])
        fc_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC')
        self.rmsprop = optimizer.minimize(cost, global_step=global_step, var_list=fc_variables)

        # masks = tf.Variable(np.zeros((50, 512)))
        # o3_reshaped = tf.reshape(o3, shape=[512])
        # o3_dropout_samples = tf.mul(masks, o3_reshaped)
        # q_value_samples = tf.add(tf.matmul(o3_dropout_samples, w4), b4)
        # most_optimal_preds = tf.reduce_max(q_value_samples, reduction_indices=0)
        # print "read this  " * 100, most_optimal_preds.get_shape()
