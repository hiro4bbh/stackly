import os.path
from struct import unpack
from stackly import xpy
from stackly.dataset import Dataset

class MNISTDataset(Dataset):
    SERVERURL = 'http://yann.lecun.com/exdb/mnist/'
    FILENAMES = {
        'train-images': 'train-images-idx3-ubyte.gz',
        'train-labels': 'train-labels-idx1-ubyte.gz',
        'test-images': 't10k-images-idx3-ubyte.gz',
        'test-labels': 't10k-labels-idx1-ubyte.gz',
    }
    def __init__(self, path):
        super(MNISTDataset, self).__init__(path)
        self.data = {}
        for key in self.FILENAMES:
            filename = self.FILENAMES[key]
            self.download_file(os.path.join(self.SERVERURL, filename))
            self.decompress_file(filename)
            filename, _ = self.detect_compression(filename)
            filedata = self.read_file(filename)
            if len(filedata) < 8:
                raise Exception('illegal header for file {}'.format(filename))
            magic, nitems = unpack('>II', filedata[:8])
            p = 8
            if key.endswith('images'):
                print('reading {} image(s) in {}'.format(nitems, filename))
                if not magic == 0x00000803:
                    raise Exception('illegal magic for image file {}'.format(filename))
                if len(filedata) - p < 8:
                    raise Exception('illegal header for image file {}'.format(filename))
                nrows, ncols = unpack('>II', filedata[p:p+8])
                p += 8
                if not len(filedata) - p == nitems*nrows*ncols:
                    raise Exception('malformed image file {}'.format(filename))
                self.data[key] = xpy.array(bytearray(filedata[p:]), dtype=xpy.float16).reshape(nitems, nrows, ncols)
            else:
                print('reading {} label(s) in {}'.format(nitems, filename))
                if not magic == 0x00000801:
                    raise Exception('illegal magic for label file {}'.format(filename))
                if not nitems == len(filedata) - p:
                    raise Exception('malformed label file {}'.format(filename))
                self.data[key] = xpy.array(bytearray(filedata[p:]))
