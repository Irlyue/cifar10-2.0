import my_utils as mu
import tensorflow as tf

from experiment import Experiment

parser = mu.get_default_parser()


def main():
    experiment = Experiment(config)
    experiment.eval()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    config = mu.load_config()
    main()
