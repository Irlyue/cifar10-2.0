import tensorflow as tf
import tensorflow.contrib.slim as slim


def default_arg_scope(weight_decay=0.0001,
                      batch_norm_decay=0.997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True,
                      activation_fn=tf.nn.relu,
                      use_batch_norm=True):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      activation_fn: The activation function which is used in ResNet.
      use_batch_norm: Whether or not to use batch normalization.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': None,
        'fused': True,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def res_block(x, filters=(16, 16), scope='block', repeat=1):
    def _base_op(inp):
        conv1 = slim.conv2d(inp, filters[0], 3)
        conv2 = slim.conv2d(conv1, filters[1], 3)
        if inp.shape[-1] != filters[-1]:
            inp = slim.conv2d(inp, filters[-1], 1)
        conv = inp + conv2
        return conv
    with tf.variable_scope(scope):
        ret = x
        for i in range(repeat):
            ret = _base_op(ret)
        return ret


def delete_if_exists(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)

