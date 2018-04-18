import tensorflow as tf
import my_utils as mu


logger = mu.get_default_logger()


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, ckpt_dir=None, ckpt_path=None, beta=0.99):
        self.beta = beta
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        if not (ckpt_dir or ckpt_path):
            raise ValueError('Only one of the arguments `ckpt_dir` or `ckpt_path` should be provided')
        if ckpt_path and ckpt_dir:
            logger.warning('Both `ckpt_path` and `ckpt_dir` are provided, `ckpt_path` will be used!')
        self.saver = None

    def begin(self):
        with tf.variable_scope('variable_moving_average'):
            variable_averages = tf.train.ExponentialMovingAverage(self.beta)
            self.saver = tf.train.Saver(variable_averages.variables_to_restore())

    def after_create_session(self, session, coord=None):
        ckpt_path = tf.train.latest_checkpoint(self.ckpt_dir) if self.ckpt_path is None else self.ckpt_path
        self.saver.restore(session, ckpt_path)
        logger.info('Model from %s restored', ckpt_path)