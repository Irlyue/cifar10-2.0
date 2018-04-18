import tensorflow as tf
import my_utils as mu
import tensorflow.contrib.slim as slim

from collections import OrderedDict


class Model:
    def __init__(self):
        pass

    def __call__(self, features, labels, mode, params):
        self.on_call(features, labels, mode, params)

        global_step = tf.train.get_or_create_global_step()
        logits = self.inference()
        output = tf.argmax(logits, axis=1, name='output')

        if mode == tf.estimator.ModeKeys.PREDICT:
            pass

        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
        data_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope='data_loss')
        loss = tf.add(reg_loss, data_loss, name='total_loss')
        accuracy = tf.metrics.accuracy(labels=labels, predictions=output)
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                'accuracy': accuracy
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('accuracy', accuracy[1])
            solver = tf.train.AdamOptimizer(self.params.lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = solver.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def on_call(self, feautures, labels, mode, params):
        endpoints = OrderedDict()
        endpoints['images'] = feautures
        self.endpoints = endpoints
        self.mode = mode
        self.params = params

    def inference(self):
        images = self.endpoints['images']
        with slim.arg_scope(mu.default_arg_scope(weight_decay=self.params.reg)):
            # with slim.arg_scope([slim.batch_norm], is_training=True):
            with slim.arg_scope([slim.batch_norm], is_training=(self.mode == tf.estimator.ModeKeys.TRAIN)):
                net = slim.conv2d(images, 16, 5, scope='conv0')
                net = slim.max_pool2d(net, 3, 2, scope='pool0')
                net = mu.res_block(net, filters=[32, 32], scope='block1', repeat=3)
                net = slim.max_pool2d(net, 2, 2, scope='pool1')
                net = mu.res_block(net, filters=[64, 64], scope='block2', repeat=4)
                net = slim.max_pool2d(net, 2, 2, scope='pool2')
                net = mu.res_block(net, filters=[128, 128], scope='block3', repeat=4)
                net = mu.res_block(net, filters=[256, 256], scope='block4', repeat=3)
                net = tf.reduce_mean(net, axis=(1, 2), name='GAP')
                net = slim.fully_connected(net, self.params.n_classes, activation_fn=None)
                return net
