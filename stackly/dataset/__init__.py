import os

class Dataset:
    def __init__(self, path):
        self.path = path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
    def download_file(self, url, filename=None, cache=True):
        if filename is None:
            filename = os.path.basename(url)
        filename = os.path.join(self.path, filename)
        if cache:
            if os.path.exists(filename):
                return
        print('downloading {} ...'.format(url))
        from urllib.request import urlretrieve
        return urlretrieve(url, filename=filename)[0]
    def detect_compression(self, filename):
        if filename.endswith('.gz'):
            return filename[:-3], 'gzip'
        return filename, None
    def decompress_file(self, filename, cache=True):
        filename = os.path.join(self.path, filename)
        dfilename, dtype = self.detect_compression(filename)
        if dtype is None:
            return
        if cache:
            if os.path.exists(dfilename):
                return
        print('decompressing {} (type: {}) ...'.format(filename, dtype))
        if dtype == 'gzip':
            import gzip
            with gzip.open(filename) as df:
                with open(dfilename, 'wb') as f:
                    f.write(df.read())
    def read_file(self, filename):
        with open(os.path.join(self.path, filename), 'rb') as f:
            return f.read()

from stackly.dataset.mnist import *
