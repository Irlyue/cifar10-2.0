import sys
import tarfile
import my_utils as mu
import urllib.request as request


def maybe_download_and_extract():
    dest_dir = '/home/wenfeng/datasets/cifar10'
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

    if not mu.path_exists(dest_dir):
        mu.mkdirs(dest_dir)

    filename = data_url.split('/')[-1]
    filepath = mu.path_join(dest_dir, filename)
    if not mu.path_exists(filepath):
        print('{} doesn\'t exist and will be downloaded!'.format(filename))

        def _progress(count, block_size, total_size):
            fraction = 100.0 * count * block_size / total_size
            sys.stdout.write('\r>> Downloading {} {:.1f}%'.format(filename, fraction))
            sys.stdout.flush()

        request.urlretrieve(data_url, filepath, reporthook=_progress)
        print()
        print('Successfully downloaded {}'.format(filename))

    extracted_dir = mu.path_join(dest_dir, 'cifar-10-batches-bin')
    if not mu.path_exists(extracted_dir):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)


if __name__ == '__main__':
    maybe_download_and_extract()
