import my_utils as mu
import tensorflow as tf

from models.model import Model
from inputs.data_gen import Cifar10InputFunction


logger = mu.get_default_logger()


class Experiment:
    def __init__(self, config):
        self.config = config
        self.mode = ''
        self.__estimator = None

    def get_input_fn(self):
        config = self.config
        input_fn = Cifar10InputFunction(type=self.mode,
                                        batch_size=config.batch_size,
                                        n_epochs=config.n_epochs,
                                        base_dir=config.data_dir)
        return input_fn

    def predict(self):
        self.switch_to_predict()
        return self.estimator.predict(self.get_input_fn())

    def train(self):
        self.switch_to_train()

        if self.config.delete:
            logger.info('Deleting existing checkpoint files...')
            mu.delete_if_exists(self.config.model_dir)

        self.estimator.train(self.get_input_fn())

    def eval(self, ckpt_path=None):
        self.switch_to_eval()
        with mu.Timer() as timer:
            result = self.estimator.evaluate(self.get_input_fn(), checkpoint_path=ckpt_path)

        result['data'] = self.mode
        logger.info('Done in %.fs', timer.eclipsed)
        logger.info('\n%s%s%s\n', '*'*10, result, '*'*10)

    def switch_to_train(self):
        self.mode = 'train'

    def switch_to_eval(self):
        self.mode = 'eval'

    def switch_to_predict(self):
        self.mode = 'predict'

    @property
    def estimator(self):
        if self.__estimator is not None:
            return self.__estimator
        model_fn = Model()
        run_config = mu.load_run_config()
        est = tf.estimator.Estimator(model_fn, model_dir=self.config.model_dir, params=self.config, config=run_config)
        self.__estimator = est
        return self.__estimator

    @estimator.setter
    def estimator(self, value):
        self.__estimator = value
