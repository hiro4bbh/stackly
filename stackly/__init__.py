import os
import sys

import numpy
xpy = sys.modules['numpy']

try:
    import cupy
    if not 'FORCE_NUMPY' in os.environ:
        xpy = sys.modules['cupy']
        memory_pool = cupy.cuda.MemoryPool()
        cupy.cuda.set_allocator(memory_pool.malloc)
        pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
        cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
except:
    pass

def asnumpy(x):
    if xpy == numpy:
        return x
    else:
        return cupy.asnumpy(x)

def asxpy(x):
    if xpy == numpy:
        return x
    else:
        return cupy.asarray(x)

def normalize(X, axis=0, ddof=1):
    dtype = xpy.float64
    if X.dtype == xpy.float16:
        dtype = xpy.float32
    Xmean = xpy.mean(X, axis=axis)
    Xdev = X - Xmean
    Xvar = xpy.sum(xpy.power(Xdev, 2, dtype=dtype), axis=axis)/(X.shape[axis] - ddof)
    return xpy.multiply(Xdev, xpy.where(Xvar == 0, 0, xpy.power(Xvar, -0.5)), dtype=X.dtype)

class Layer:
    def get_params(self):
        raise NotImplementedError()
    def get_prev_layers(self):
        raise NotImplementedError()
    def get_shape(self):
        raise NotImplementedError()
    def get_size(self):
        return int(numpy.prod(self.get_shape()))
    def get_param_shape(self):
        return (0,)
    def get_param_size(self):
        return xpy.prod(self.get_param_shape())
    def get_dtype(self):
        raise NotImplementedError()
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, x, y, dy, m):
        raise NotImplementedError()
    def __repr__(self):
        return "{}({})".format(type(self).__name__, ','.join([str(param) for param in self.get_params()]))

class Constant(Layer):
    def __init__(self, value):
        self.value = value
    def get_params(self):
        return (self.value,)
    def get_prev_layers(self):
        return ()
    def get_shape(self):
        return self.value.shape
    def get_dtype(self):
        return self.value.dtype
    def forward(self, x):
        return asxpy(numpy.array([self.value], dtype=self.value.dtype))

class Variable(Layer):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def get_params(self):
        return (self.name, self.shape, self.dtype)
    def get_prev_layers(self):
        return ()
    def get_shape(self):
        return self.shape
    def get_dtype(self):
        return self.dtype

class Concat(Layer):
    def __init__(self, x):
        self.x = x
        self.indices = numpy.cumsum([v.get_size() for v in x])
        self.shape = self.indices[-1]
        self.indices = self.indices[:-1]
    def get_params(self):
        return (self.x,)
    def get_prev_layers(self):
        return self.x
    def get_shape(self):
        return (self.shape,)
    def get_dtype(self):
        return self.x[0].get_dtype()
    def forward(self, x):
        sizes = xpy.array([v.shape[0] for v in x])
        repeats = xpy.max(sizes)//sizes
        return xpy.concatenate([x[i].repeat(int(repeats[i]), axis=0) for i in range(len(x))], axis=1)
    def backward(self, x, y, dy, dw):
        dx = xpy.hsplit(dy, self.indices)
        return dx

class FullyConnected(Layer):
    def __init__(self, x, l, mutable=True):
        self.x = x
        self.w = asxpy(numpy.random.normal(size=l*x.get_size()).astype(x.get_dtype()).reshape(l, x.get_size()))
        self.mutable = mutable
    def get_params(self):
        return (self.x, self.w.shape[0], self.mutable)
    def get_prev_layers(self):
        return (self.x,)
    def get_shape(self):
        return (self.w.shape[0],)
    def get_param_shape(self):
        return self.w.shape
    def get_dtype(self):
        return self.w.dtype
    def forward(self, x):
        x = x[0].reshape(x[0].shape[0], -1)
        return xpy.tensordot(x, self.w, axes=([1], [1]))
    def backward(self, x, y, dy, m):
        x = x[0].reshape(x[0].shape[0], -1)
        dx = xpy.tensordot(dy, self.w, axes=([1], [0]))
        if self.mutable:
            dw = xpy.tensordot(dy, x, axes=([0], [0]))/dy.shape[0]
            self.w += m(dw)
        return (dx,)

class ReLU(Layer):
    def __init__(self, x):
        self.x = x
    def get_params(self):
        return (self.x,)
    def get_prev_layers(self):
        return(self.x,)
    def get_shape(self):
        return self.x.get_shape()
    def get_dtype(self):
        return self.x.get_dtype()
    def forward(self, x):
        return xpy.maximum(x[0], 0.0)
    def backward(self, x, y, dy, m):
        return (dy*xpy.sign(y),)

class SquaredLoss:
    if not xpy == numpy:
        loss_kernel = xpy.ReductionKernel(
                'T y, T z',
                'T l',
                '(y - z) * (y - z)',
                'a + b',
                'l = a',
                '0',
                'squared_loss_loss_kernel'
            )
        dloss_kernel = xpy.ElementwiseKernel(
                'T y, T z',
                'T d',
                'd = -2 * (y - z);',
                'squared_loss_dloss_kernel'
            )
    def loss(y, z):
        if xpy == numpy:
            return float(xpy.sum((y - z)**2, dtype=xpy.float64))
        else:
            return float(SquaredLoss.loss_kernel(y, z))
    def dloss(y, z):
        if xpy == numpy:
            return -2*(y - z)
        else:
            return SquaredLoss.dloss_kernel(y, z)

class NegativeSoftmaxCrossEntropyLoss:
    def loss(y, z):
        z = z.astype(xpy.float64)
        zmax = xpy.amax(z, axis=1)
        z_zmax = z - zmax[:,xpy.newaxis]
        loss = xpy.sum(xpy.log(xpy.sum(xpy.exp(z_zmax), axis=1))) + xpy.sum(zmax)
        for i in range(z.shape[0]):
            loss -= z[i, int(y[i])]
        return float(loss)
    def dloss(y, z):
        zmax = xpy.amax(z, axis=1)
        exp_z_zmax = xpy.exp(z - zmax[:,xpy.newaxis])
        p = exp_z_zmax/xpy.sum(exp_z_zmax, axis=1)[:,xpy.newaxis]
        for i in range(z.shape[0]):
            p[i, int(y[i])] -= 1
        return p

class OptimizerBase:
    def __init__(self, fn, loss):
        self.fn = fn
        self.loss = loss
        self.layers = []
        self.prev_layers = []
        self.param_mans = []
        def traverse_layers(layer):
            layer_id = len(self.layers)
            self.layers.append(layer)
            prev_layers = layer.get_prev_layers()
            self.prev_layers.append([])
            param_man = None
            if len(prev_layers) > 0:
                param_man = self.ParameterManager(self, layer.get_param_shape(), layer.get_dtype())
            self.param_mans.append(param_man)
            for prev_layer in prev_layers:
                self.prev_layers[layer_id].append(traverse_layers(prev_layer))
            return layer_id
        traverse_layers(self.fn)
    def param_man(self, size):
        raise NotImplementedError
    def forward(self, x):
        h = [None]*len(self.layers)
        def forward_layer(layer_id):
            layer = self.layers[layer_id]
            prev_layers = self.prev_layers[layer_id]
            if isinstance(layer, Variable):
                h[layer_id] = x[layer.name]
            else:
                for prev_layer_id in prev_layers:
                    forward_layer(prev_layer_id)
                h[layer_id] = layer.forward(list(map(h.__getitem__, prev_layers)))
        forward_layer(0)
        return h
    def backward(self, x, h, dloss_val):
        def backward_layer(layer_id, dy):
            prev_layers = self.prev_layers[layer_id]
            if len(prev_layers) > 0:
                dx = self.layers[layer_id].backward(list(map(h.__getitem__, prev_layers)), h[layer_id], dy, self.param_mans[layer_id])
                for j in range(len(dx)):
                    backward_layer(prev_layers[j], dx[j])
        backward_layer(0, dloss_val)
    def run(self, x, y):
        h = self.forward(x)
        l = self.loss.loss(y, h[0])/y.shape[0]
        self.backward(x, h, self.loss.dloss(y, h[0]))
        return l

class Adam(OptimizerBase):
    if not xpy == numpy:
        kernel = xpy.ElementwiseKernel(
                'T t, T alpha, T beta1, T beta2, T epsilon, T g',
                'S m, S v, S delta',
                '''
                    m = beta1 * m + (1 - beta1) * g;
                    v = beta2 * v + (1 - beta2) * pow(g, 2);
                    delta = -alpha * (m/(1 - pow(beta1, t)))/(sqrt(v / (1 - pow(beta2, t))) + epsilon);
                ''',
                'adam_kernel'
            )
    def __init__(self, fn, loss, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(fn, loss)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    class ParameterManager:
        def __init__(self, optim, shape, dtype):
            self.optim = optim
            self.t = 0
            mvdtype = xpy.float64
            if dtype == xpy.float16:
                mvdtype = xpy.float32
            self.m = xpy.zeros(shape, dtype=mvdtype)
            self.v = xpy.zeros(shape, dtype=mvdtype)
        def __call__(self, g):
            t = self.t = self.t + 1
            if xpy == numpy:
                m = self.m = self.optim.beta1*self.m + (1 - self.optim.beta1)*g
                v = self.v = self.optim.beta2*self.v + (1 - self.optim.beta2)*g**2
                delta = -self.optim.alpha*(m/(1 - self.optim.beta1**t))/((v/(1 - self.optim.beta2**t))**0.5 + self.optim.epsilon)
            else:
                delta = xpy.zeros(self.m.shape, dtype=self.m.dtype)
                delta = Adam.kernel(t, self.optim.alpha, self.optim.beta1, self.optim.beta2, self.optim.epsilon, g, self.m, self.v, delta)[2]
            return delta.astype(g.dtype)

class Dataset:
    def __init__(self, path):
        self.path = path
        import os
        from os import path
        if not path.isdir(self.path):
            os.makedirs(self.path)
    def download_file(self, url, filename=None, cache=True):
        from os import path
        if filename is None:
            filename = path.basename(url)
        filename = path.join(self.path, filename)
        if cache:
            if path.exists(filename):
                return
        print('downloading {} ...', url)
        from urllib.request import urlretrieve
        return urlretrieve(url, filename=filename)[0]
    def detect_compression(self, filename):
        if filename.endswith('.gz'):
            return filename[:-3], 'gzip'
        return filename, None
    def decompress_file(self, filename, cache=True):
        from os import path
        filename = path.join(self.path, filename)
        dfilename, dtype = self.detect_compression(filename)
        if dtype is None:
            return
        if cache:
            if path.exists(dfilename):
                return
        print('decompressing {} (type: {}) ...'.format(filename, dtype))
        if dtype == 'gzip':
            import gzip
            with gzip.open(filename) as df:
                with open(dfilename, 'wb') as f:
                    f.write(df.read())
    def read_file(self, filename):
        from os import path
        with open(path.join(self.path, filename), 'rb') as f:
            return f.read()

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
        from os import path
        from struct import unpack
        for key in self.FILENAMES:
            filename = self.FILENAMES[key]
            self.download_file(path.join(self.SERVERURL, filename))
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
