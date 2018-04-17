import argparse

ARGUMENTS = []


class Argument:
    parser = None

    def __init__(self, *args, **kwargs):
        if 'default' in kwargs and 'help' in kwargs:
            kwargs['help'] = '{}(default {!r})'.format(kwargs['help'], kwargs['default'])
        self.parser.add_argument(*args, **kwargs)

    @staticmethod
    def set_parser(parser):
        Argument.parser = parser

    @staticmethod
    def get_parser():
        return Argument.parser


def add_arg(func):
    ARGUMENTS.append(func)
    return func


#####################################
#       Customized Arguments        #
#####################################
@add_arg
def eval_data():
    Argument('--eval_data', default='eval', type=str, help='data to evaluate')


@add_arg
def learning_rate():
    Argument('--lr', default=1e-3, type=float, help='learning rate')


@add_arg
def data_dir():
    Argument('--data_dir', default='/home/wenfeng/datasets/cifar10', type=str,
             help='data directory')


@add_arg
def solver():
    Argument('--solver', default='adam', type=str,
             help='solver, choose from(adam, sgd)')


@add_arg
def image_size():
    Argument('-s', '--image_size', default=[224, 224], type=int, nargs=2,
             help='input image size')


@add_arg
def delete():
    Argument('-d', '--delete', action='store_true',
             help='delete checkpoint files and train from scratch')


@add_arg
def model_dir():
    Argument('--model_dir', default='/tmp/cifar10', type=str,
             help='model directory')


@add_arg
def n_epochs():
    Argument('--n_epochs', default=1, type=int,
             help='number of epochs for the input function')


@add_arg
def batch_size():
    Argument('--batch_size', default=32, type=int,
             help='batch size')


@add_arg
def n_classes():
    Argument('--n_classes', default=10, type=int,
             help='number of classes')


def get_default_parser():
    Argument.set_parser(argparse.ArgumentParser())
    for cls in ARGUMENTS:
        cls()
    return Argument.get_parser()


if __name__ == '__main__':
    parser = get_default_parser()
    print(parser.parse_args())
