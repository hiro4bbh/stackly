import argparse
import sys

parser = argparse.ArgumentParser(description='stackly test code')
parser.add_argument('--model', default='3FC')
args = parser.parse_args()
print('args: {}'.format(args))

import numpy
from stackly import xpy, normalize, Constant, Variable, Concat, FullyConnected, SpatialConvolution, MaxPooling2D, ReLU, Dropout, SquaredLoss, NegativeSoftmaxCrossEntropyLoss, Adam
from stackly.dataset import MNISTDataset

mnist = MNISTDataset('data/mnist')

numpy.random.seed(0)
xpy.random.seed(0)

x = Variable('image', (28, 28), dtype=xpy.float32)
if args.model == '1FC':
    y = FullyConnected(x, 10)
    # Least Precision:
    #   t= 175: loss=0.22811
    #   train: 56330/60000, test: 9242/10000
elif args.model == '3FC':
    y1 = FullyConnected(x, 320)
    y2 = ReLU(y1)
    y2 = FullyConnected(y2, 50)
    y3 = ReLU(y2)
    y = FullyConnected(y3, 10)
elif args.model == '1SC':
    y1 = SpatialConvolution(x, 8, (1, 8, 8), (4, 4))
    y2 = ReLU(y1)
    y = FullyConnected(y2, 10)
elif args.model == '1DSC':
    y1 = SpatialConvolution(x, 8, (1, 8, 8), (4, 4))
    y2 = ReLU(y1)
    y2 = Dropout(y2)
    y = FullyConnected(y2, 10)
elif args.model == '1MP1SC':
    y1 = SpatialConvolution(x, 8, (1, 8, 8), (4, 4))
    y2 = ReLU(y1)
    y2 = MaxPooling2D(y2, (3, 3), (1, 1))
    y2 = ReLU(y2)
    y = FullyConnected(y2, 10)
else:
    raise Exception('unknown model: {}'.format(args.model))

print("model {}: {}".format(args.model, y))

optim = Adam(y, NegativeSoftmaxCrossEntropyLoss)

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
for t in range(1, 200):
    p = xpy.random.permutation(images.shape[0])
    images, labels = images[p], labels[p]
    loss = 0.0
    for batch in range(nbatches):
        start, end = batch*batch_size, (batch+1)*batch_size
        batch_loss = optim.run({'image': images[start:end]}, labels[start:end])
        #print("t={:4d}({:.2f}%): batch_loss={:.5f}; {}, {}, {}".format(t, batch*100/nbatches, batch_loss, xpy.linalg.norm(y1.w), xpy.linalg.norm(y2.w), xpy.linalg.norm(y3.w)))
        loss += batch_loss
        sys.stdout.write('.')
        sys.stdout.flush()
    loss /= nbatches
    print("t={:4d}: loss={:.5f}".format(t, loss))
    train_predicteds = optim.forward({'image': images}, training=False)[0]
    train_correcteds = labels == xpy.argmax(train_predicteds, axis=1)
    test_predicteds = optim.forward({'image': test_images}, training=False)[0]
    test_correcteds = test_labels == xpy.argmax(test_predicteds, axis=1)
    print("train: {}/{}, test: {}/{}".format(xpy.sum(train_correcteds), len(train_correcteds), xpy.sum(test_correcteds), len(test_correcteds)))
