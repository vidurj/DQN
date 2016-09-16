import numpy as np
import tensorflow as tf
import cv2


class DQN:
    def __init__(self, params, name):
        self.network_type = 'nips'
        self.params = params
        self.network_name = name
        self.x = tf.placeholder('float32', [None, 84, 84, 4], name=self.network_name + '_x')
        self.q_t = tf.placeholder('float32', [None], name=self.network_name + '_q_t')
        self.actions = tf.placeholder("float32", [None, params['num_act']], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float32", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float32", [None], name=self.network_name + '_terminals')

        # conv1
        with tf.variable_scope("CONV"):
            layer_name = 'conv1';
            size = 8;
            channels = 4;
            filters = 16;
            stride = 4
            self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                                  name=self.network_name + '_' + layer_name + '_weights')
            self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
            self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='VALID',
                                   name=self.network_name + '_' + layer_name + '_convs')
            self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name + '_' + layer_name + '_activations')
            # self.n1 = tf.nn.lrn(self.o1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # conv2
            layer_name = 'conv2';
            size = 4;
            channels = 16;
            filters = 32;
            stride = 2
            self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                                  name=self.network_name + '_' + layer_name + '_weights')
            self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
            self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='VALID',
                                   name=self.network_name + '_' + layer_name + '_convs')
            self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), name=self.network_name + '_' + layer_name + '_activations')
            # self.n2 = tf.nn.lrn(self.o2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # flat
            o2_shape = self.o2.get_shape().as_list()

        with tf.variable_scope("FC"):
            self.mask = tf.placeholder(tf.float32, shape=512)

            # fc3
            layer_name = 'fc3';
            hiddens = 512;
            dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
            self.o2_flat = tf.reshape(self.o2, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
            self.w3 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                                  name=self.network_name + '_' + layer_name + '_weights')
            self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
            self.ip3 = tf.add(tf.matmul(self.o2_flat, self.w3), self.b3, name=self.network_name + '_' + layer_name + '_ips')
            self.o3 = tf.nn.relu(self.ip3, name=self.network_name + '_' + layer_name + '_activations')
            self.o3_dropout = self.o3 * self.mask

            # fc4
            layer_name = 'fc4';
            hiddens = params['num_act'];
            dim = 512
            self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                                  name=self.network_name + '_' + layer_name + '_weights')
            self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
            self.y = tf.add(tf.matmul(self.o3, self.w4), self.b4, name=self.network_name + '_' + layer_name + '_outputs')
            # dummy for nature network
            self.w5 = tf.Variable(tf.constant(1.0))
            self.b5 = tf.Variable(tf.constant(1.0))
            # Q,Cost,Optimizer
            self.discount = tf.constant(self.params['discount'])
            self.yj = tf.add(self.rewards, tf.mul(1.0 - self.terminals, tf.mul(self.discount, self.q_t)))
            self.Qxa = tf.mul(self.y, self.actions)
            self.Q_pred = tf.reduce_sum(self.Qxa, reduction_indices=1)
            # self.yjr = tf.reshape(self.yj,(-1,1))
            # self.yjtile = tf.concat(1,[self.yjr,self.yjr,self.yjr,self.yjr])
            # self.yjax = tf.mul(self.yjtile,self.actions)

            # half = tf.constant(0.5)
            self.diff = tf.sub(self.yj, self.Q_pred)

        self.diff_square = tf.mul(tf.constant(0.5), tf.pow(self.diff, 2))

        if self.params['batch_accumulator'] == 'sum':
            self.cost = tf.reduce_sum(self.diff_square)
        else:
            self.cost = tf.reduce_mean(self.diff_square)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(self.params['lr'], self.params['rms_decay'], 0.0,
                                                 self.params['rms_eps'])
        fc_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='FC')
        self.rmsprop = optimizer.minimize(self.cost, global_step=self.global_step, var_list=fc_variables)

        print "READ THIS" * 30, self.o3.get_shape()
        self.masks = tf.placeholder(tf.float32, (None, 512))
        o3_reshaped = tf.reshape(self.o3, shape=[512])
        print o3_reshaped.get_shape(), self.masks.get_shape()
        self.o3_dropout_samples = tf.mul(self.masks, o3_reshaped)
        print self.o3_dropout_samples.get_shape()
        self.q_value_samples = tf.add(tf.matmul(self.o3_dropout_samples, self.w4), self.b4)
        print self.q_value_samples.get_shape()
        best_foot = tf.reduce_max(self.q_value_samples, reduction_indices=1)
        print best_foot.get_shape()
        self.best_mask = tf.arg_max(best_foot, dimension=0)
        print self.best_mask.get_shape()
