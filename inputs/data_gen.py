import my_utils as mu
import tensorflow as tf

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3


class Cifar10InputFunction:
    def __init__(self, type='train', batch_size=32, n_epochs=1,
                 base_dir='/tmp/cifar10'):
        self.type= type
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        fmt = mu.path_join(base_dir, 'cifar-10-batches-bin/data_batch_{}.bin')
        self.filenames = [fmt.format(i) for i in range(1, 6)]
        if self.type == 'eval':
            self.filenames = [mu.path_join(base_dir, 'cifar-10-batches-bin/test_batch.bin')]

    def __call__(self):
        label_bytes = 1
        image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
        record_bytes = label_bytes + image_bytes
        data = tf.data.FixedLengthRecordDataset(filenames=self.filenames,
                                                record_bytes=record_bytes)

        is_training = (self.type == 'train')
        if is_training:
            data.shuffle(self.batch_size * 4)

        data = data.map(self._parse_record)
        data = data.map(lambda image, label: (self._preprocess_image(image, is_training), label))
        data = data.repeat(self.n_epochs)
        data = data.prefetch(self.batch_size * 4)
        data = data.batch(self.batch_size)
        images, labels = data.make_one_shot_iterator().get_next()
        return images, labels

    @staticmethod
    def _parse_record(record):
        label_bytes = 1
        image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
        record_bytes = label_bytes + image_bytes

        record_vector = tf.decode_raw(record, tf.uint8)

        label = tf.cast(record_vector[0], tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            record_vector[label_bytes:record_bytes], [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])

        # Convert from [depth, height, width] to [height, width, depth], and cast as
        # float32.
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
        return image, label

    @staticmethod
    def _preprocess_image(image, is_training=False):
        if is_training:
            # Resize the image to add four extra pixels on each side.
            image = tf.image.resize_image_with_crop_or_pad(
                image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

            # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
            image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)
        return image
