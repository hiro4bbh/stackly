import numpy
from stackly import xpy, normalize, Constant, Variable, Concat, FullyConnected, ReLU, SquaredLoss, NegativeSoftmaxCrossEntropyLoss, Adam
from stackly import MNISTDataset

mnist = MNISTDataset('data/mnist')

xpy.random.seed(0)

x = Variable('image', (28, 28), dtype=xpy.float32)
y1 = FullyConnected(x, 320)
y2 = ReLU(y1)
y2 = FullyConnected(y2, 50)
y3 = ReLU(y2)
y3 = FullyConnected(y3, 10)

import code
code.interact(local=locals())
exit(1)

optim = Adam(y3, NegativeSoftmaxCrossEntropyLoss)

images = mnist.data['train-images']
#images = normalize(images)
images = images/255.0
labels = mnist.data['train-labels']

test_images = mnist.data['test-images']
#test_images = normalize(test_images)
test_images = test_images/255.0
test_labels = mnist.data['test-labels']

batch_size = 100
nbatches = images.shape[0]//batch_size
for t in range(1, 100):
    p = xpy.random.permutation(images.shape[0])
    images, labels = images[p], labels[p]
    loss = 0.0
    for batch in range(nbatches):
        start, end = batch*batch_size, (batch+1)*batch_size
        batch_loss = optim.run({'image': images[start:end]}, labels[start:end])
        #print("t={:4d}({:.2f}%): batch_loss={:.5f}; {}, {}, {}".format(t, batch*100/nbatches, batch_loss, xpy.linalg.norm(y1.w), xpy.linalg.norm(y2.w), xpy.linalg.norm(y3.w)))
        loss += batch_loss
    loss /= nbatches
    print("t={:4d}: loss={:.5f}".format(t, loss))
    y = optim.forward({'image': test_images})[0]
    flags = test_labels == xpy.amax(y, axis=1)
    print("{}/{}".format(xpy.sum(flags), len(flags)))

"""
import time
start_time = time.time()
xpy.random.seed(0)
x = Variable('x', (3,), dtype=xpy.float16)
#b = Constant(xpy.array([1], dtype=xpy.float16))
#y1 = FullyConnected(Concat((b, x)), 3)
#y2 = ReLU(y1)
#y2 = FullyConnected(Concat((b, y2)), 1)
y = FullyConnected(x, 1)

optim = Adam(y, SquaredLoss)

beta = xpy.array([1, 2, 3], dtype=xpy.float16).reshape(1, -1)
data_x = normalize(xpy.arange(-500*3, 500*3, dtype=xpy.float16).reshape(1000, -1))
#data_y = xpy.abs(1 + xpy.tensordot(data_x, beta, axes=([1], [1])))
data_y = xpy.tensordot(data_x, beta, axes=([1], [1]))

batch_size = 100
nbatches = data_x.shape[0]//batch_size
for t in range(1, 1001):
    p = numpy.random.permutation(data_x.shape[0])
    data_x, data_y = data_x[p], data_y[p]
    loss = 0.0
    for batch in range(nbatches):
        start, end = batch*batch_size, (batch+1)*batch_size
        batch_loss = optim.run({'x': data_x[start:end]}, data_y[start:end])
        loss += batch_loss
    loss /= nbatches
    #print("t={:4d}: loss={:.5f}: W={}".format(t, loss, y.w))
print(time.time() - start_time)
"""
