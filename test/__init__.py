import unittest
import numpy.testing

FLOAT32_COARSE_PRECISION = 2
FLOAT32_FINE_PRECISION = 4

class TestPreprocessing(unittest.TestCase):
    def test_normalize(self):
        from stackly import xpy, asnumpy, normalize
        X = xpy.array([
            [-1, 9, 1],
            [1, 1, 1],
            [3, -7, 1],
        ], dtype=xpy.float32)
        Xnormalized = xpy.array([
            [-1, 1, 0],
            [0, 0, 0],
            [1, -1, 0],
        ], dtype=xpy.float32)
        numpy.testing.assert_almost_equal(asnumpy(normalize(X)), asnumpy(Xnormalized), decimal=FLOAT32_FINE_PRECISION)

class TestLayer(unittest.TestCase):
    def test_spatial_convolution_forward(self):
        import numpy
        from stackly import xpy, asnumpy, asxpy, Variable, SpatialConvolution
        image = Variable('image', (3, 9, 15), dtype=xpy.float32)
        filter_layer = SpatialConvolution(image, 4, (3, 3, 5), (3, 5))
        # Initialize the weight manually.
        w = numpy.array([numpy.repeat([[
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ]], 3, axis=0)], dtype=numpy.float32)
        filter_layer.w = asxpy(w)
        # Prepare 4 images for testing the above 4 kernels.
        images = numpy.zeros((4, 3, 9, 15), dtype=numpy.float32)
        for n in range(images.shape[0]):
            images[n] = numpy.tile(w[n], (1, 3, 3))
        images = asxpy(images)
        # Test forward.
        filtered_images = filter_layer.forward((images,))
        expected_template = numpy.full(numpy.prod(filtered_images.shape[1:]), 3, dtype=numpy.float32)
        for n in range(filtered_images.shape[0]):
            filtered_image = filtered_images[n]
            expected = numpy.copy(expected_template)
            start = n*numpy.prod(filtered_image.shape[1:])
            expected[start:(start+numpy.prod(filtered_image.shape[1:]))] = numpy.sum(w[n])
            expected = expected.reshape(filtered_image.shape)
            numpy.testing.assert_equal(asnumpy(filtered_image), expected)
    def test_max_pooling_2d_forward_backward(self):
        import numpy
        from stackly import xpy, asnumpy, asxpy, Variable, MaxPooling2D
        x = Variable('x', (3, 4, 3), dtype=xpy.float32)
        y = MaxPooling2D(x, (2, 2), (2, 1))
        data_x = numpy.zeros((1, 3, 4, 3), dtype=numpy.float32)
        data_x[0, :, 2, 0] = data_x[0, :, 0, 1] = 1.0
        data_x = asxpy(data_x)
        data_y = y.forward((data_x,))
        data_y_expected = xpy.array([[[[1, 1], [1, 0]]]], dtype=xpy.float32).repeat(3, axis=1)
        numpy.testing.assert_equal(data_y, data_y_expected)
        ddata_y = asxpy(numpy.tile(numpy.arange(1, 1+numpy.prod(data_y.shape[2:]), dtype=xpy.float32), data_y.shape[1]).reshape(data_y.shape))
        ddata_x = y.backward((data_x,), data_y, ddata_y, None)[0]
        ddata_x_expected = xpy.array([[[[0, 3, 0], [0, 0, 0], [3, 4, 4], [0, 4, 4]]]], dtype=xpy.float32).repeat(3, axis=1)
        numpy.testing.assert_equal(ddata_x, ddata_x_expected)

class TestNetworkFit(unittest.TestCase):
    def test_multivariate_linear_complete_fit(self):
        from stackly import xpy, normalize, Variable, FullyConnected, SquaredLoss, Adam
        numpy.random.seed(0)
        xpy.random.seed(0)
        x = Variable('x', (3,), dtype=xpy.float32)
        y = FullyConnected(x, 1)
        optim = Adam(y, SquaredLoss)
        self.assertEqual(str(y), 'FullyConnected(Variable(x,(3,),<class \'numpy.float32\'>),1,True)')
        beta = xpy.array([1, 2, 3], dtype=xpy.float32).reshape(1, -1)
        n = 1000
        data_x = normalize(xpy.arange(n*beta.size, dtype=xpy.float32).reshape(n, -1))
        data_y = xpy.tensordot(data_x, beta, axes=([1], [1]))
        batch_size = 100
        nbatches = data_x.shape[0]//batch_size
        for t in range(1, 201):
            p = numpy.random.permutation(data_x.shape[0])
            data_x, data_y = data_x[p], data_y[p]
            loss = 0.0
            for batch in range(nbatches):
                start, end = batch*batch_size, (batch+1)*batch_size
                batch_loss = optim.run({'x': data_x[start:end]}, data_y[start:end])
                loss += batch_loss
            loss /= nbatches
            #print("t={:4d}: loss={:.5f}: W={}".format(t, loss, y.w))
        self.assertAlmostEqual(loss, 0.0, places=FLOAT32_COARSE_PRECISION)
    def test_multivariate_linear_bias_complete_fit(self):
        from stackly import xpy, normalize, Constant, Variable, Concat, FullyConnected, SquaredLoss, Adam
        numpy.random.seed(0)
        xpy.random.seed(0)
        x = Variable('x', (3,), dtype=xpy.float32)
        b = Constant(xpy.array([1], dtype=xpy.float32))
        y = FullyConnected(Concat((x, b)), 1)
        self.assertEqual(str(y), 'FullyConnected(Concat((Variable(x,(3,),<class \'numpy.float32\'>), Constant([ 1.]))),1,True)')
        optim = Adam(y, SquaredLoss)
        beta = xpy.array([1, 2, 3], dtype=xpy.float32).reshape(1, -1)
        n = 1000
        data_x = normalize(xpy.arange(n*beta.size, dtype=xpy.float32).reshape(n, -1))
        data_y = 1 + xpy.tensordot(data_x, beta, axes=([1], [1]))
        batch_size = 100
        nbatches = data_x.shape[0]//batch_size
        for t in range(1, 251):
            p = numpy.random.permutation(data_x.shape[0])
            data_x, data_y = data_x[p], data_y[p]
            loss = 0.0
            for batch in range(nbatches):
                start, end = batch*batch_size, (batch+1)*batch_size
                batch_loss = optim.run({'x': data_x[start:end]}, data_y[start:end])
                loss += batch_loss
            loss /= nbatches
            #print("t={:4d}: loss={:.5f}: W={}".format(t, loss, y.w))
        self.assertAlmostEqual(loss, 0.0, places=FLOAT32_COARSE_PRECISION)
    def test_multivariate_rectified_linear_complete_fit(self):
        from stackly import xpy, normalize, Constant, Variable, Concat, ReLU, FullyConnected, SquaredLoss, Adam
        numpy.random.seed(0)
        xpy.random.seed(0)
        x = Variable('x', (3,), dtype=xpy.float32)
        b = Constant(xpy.array([1], dtype=xpy.float32))
        y1 = FullyConnected(Concat((x, b)), 3)
        y2 = ReLU(y1)
        #y2 = FullyConnected(Concat((y2, b)), 1)
        y2 = FullyConnected(y2, 1)
        self.assertEqual(str(y2), 'FullyConnected(ReLU(FullyConnected(Concat((Variable(x,(3,),<class \'numpy.float32\'>), Constant([ 1.]))),3,True)),1,True)')
        optim = Adam(y2, SquaredLoss)
        beta = xpy.array([1, 2, 3], dtype=xpy.float32).reshape(1, -1)
        n = 1000
        data_x = normalize(xpy.arange(n*beta.size, dtype=xpy.float32).reshape(n, -1))
        data_y = xpy.abs(1 + xpy.tensordot(data_x, beta, axes=([1], [1])))
        batch_size = 100
        nbatches = data_x.shape[0]//batch_size
        for t in range(1, 751):
            p = numpy.random.permutation(data_x.shape[0])
            data_x, data_y = data_x[p], data_y[p]
            loss = 0.0
            for batch in range(nbatches):
                start, end = batch*batch_size, (batch+1)*batch_size
                batch_loss = optim.run({'x': data_x[start:end]}, data_y[start:end])
                loss += batch_loss
            loss /= nbatches
            #print("t={:4d}: loss={:.5f}: W1={}, W2={}".format(t, loss, y1.w, y2.w))
        self.assertAlmostEqual(loss, 0.0, places=FLOAT32_COARSE_PRECISION)
    def test_multivariate_multiclass_logistic_complete_fit(self):
        from stackly import xpy, normalize, Constant, Variable, Concat, FullyConnected, NegativeSoftmaxCrossEntropyLoss, Adam
        numpy.random.seed(0)
        xpy.random.seed(0)
        x = Variable('x', (3,), dtype=xpy.float32)
        b = Constant(xpy.array([1], dtype=xpy.float32))
        y = FullyConnected(Concat((x, b)), 3)
        self.assertEqual(str(y), 'FullyConnected(Concat((Variable(x,(3,),<class \'numpy.float32\'>), Constant([ 1.]))),3,True)')
        optim = Adam(y, NegativeSoftmaxCrossEntropyLoss)
        nclasses = 3
        n = 300
        data_x, data_y = None, None
        for classid in range(nclasses):
            beta = xpy.array([classid*3, classid*3+1, classid*3+2], dtype=xpy.float32).reshape(1, -1)
            x_ = normalize(xpy.arange(n*beta.size, dtype=xpy.float32).reshape(n, -1))
            if data_x is None:
                data_x = x_
            else:
                data_x = xpy.concatenate((data_x, x_))
            y_ = xpy.full((1000, 1), classid)
            if data_y is None:
                data_y = y_
            else:
                data_y = xpy.concatenate((data_y, y_))
        batch_size = 30
        nbatches = data_x.shape[0]//batch_size
        for t in range(1, 251):
            p = numpy.random.permutation(data_x.shape[0])
            data_x, data_y = data_x[p], data_y[p]
            loss = 0.0
            for batch in range(nbatches):
                start, end = batch*batch_size, (batch+1)*batch_size
                batch_loss = optim.run({'x': data_x[start:end]}, data_y[start:end])
                loss += batch_loss
            loss /= nbatches
            #print("t={:4d}: loss={:.5f}: W={}".format(t, loss, y.w))
        self.assertAlmostEqual(loss, 0.0, places=FLOAT32_COARSE_PRECISION)
    def test_spatial_convolution_complete_fit(self):
        import numpy
        from stackly import xpy, asnumpy, asxpy, Variable, FullyConnected, SpatialConvolution, NegativeSoftmaxCrossEntropyLoss, Adam
        numpy.random.seed(0)
        xpy.random.seed(0)
        image = Variable('image', (3, 9, 15), dtype=xpy.float32)
        filter_layer = SpatialConvolution(image, 4, (3, 3, 5), (3, 5))
        y = FullyConnected(filter_layer, 4)
        # Initialize the population weight.
        w = numpy.array([numpy.repeat([[
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]], 3, axis=0), numpy.repeat([[
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ]], 3, axis=0)], dtype=numpy.float32)
        self.assertEqual(str(y), 'FullyConnected(SpatialConvolution(Variable(image,(3, 9, 15),<class \'numpy.float32\'>),4,(3, 3, 5),(3, 5),True),4,True)')
        optim = Adam(y, NegativeSoftmaxCrossEntropyLoss)
        # Prepare 4*250 images for testing the above 4 kernels.
        images = numpy.zeros((4*250, 3, 9, 15), dtype=numpy.float32)
        data_y = numpy.zeros(4*250)
        for t in range(w.shape[0]):
            data_y[t*250:(t+1)*250] = t
            for n in range(250):
                images[t*250+n] = numpy.tile(w[t], (1, 3, 3))
        images = asxpy(images)
        batch_size = 50
        nbatches = images.shape[0]//batch_size
        for t in range(1, 51):
            p = numpy.random.permutation(images.shape[0])
            images, data_y = images[p], data_y[p]
            loss = 0.0
            for batch in range(nbatches):
                start, end = batch*batch_size, (batch+1)*batch_size
                batch_loss = optim.run({'image': images[start:end]}, data_y[start:end])
                loss += batch_loss
            loss /= nbatches
            #print("t={:4d}: loss={:.5f}".format(t, loss))
            #predicteds = optim.forward({'image': images})[0]
            #correcteds = data_y == xpy.argmax(predicteds, axis=1)
            #print("precision: {}/{}".format(xpy.sum(correcteds), len(correcteds)))
        self.assertAlmostEqual(loss, 0.0, places=FLOAT32_COARSE_PRECISION)

def make_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestPreprocessing))
    suite.addTests(unittest.makeSuite(TestLayer))
    suite.addTests(unittest.makeSuite(TestNetworkFit))
    return suite
