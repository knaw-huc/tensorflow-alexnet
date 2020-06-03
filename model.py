import tensorflow as tf

class Model:

    def __init__(self, image_size = 224, n_classes = 16, fc_size = 1024):

        self.n_classes = n_classes
        tf.compat.v1.disable_eager_execution()  
        self.dropout = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")
        self.input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="input_images")

        ## First convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3,3,3,16],stddev=1e-1), name="conv1_weights")
        conv = tf.nn.conv2d(self.input_images, kernel, [1,2,2,1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([16]))
        conv_with_bias = tf.nn.bias_add(conv, bias)
        # Rectifier see: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        conv1 = tf.nn.leaky_relu(conv_with_bias, name="conv1")

        # local response normalization see: https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/
        lrn1 = tf.nn.lrn(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        pooled_conv1 = tf.nn.max_pool2d(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pool1")

        ## Second convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 16, 64],stddev=1e-1),
                             name="conv2_weights")
        conv = tf.nn.conv2d(pooled_conv1, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([64]), name="conv2_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.leaky_relu(conv_with_bias, name="conv2")
        lrn2 = tf.nn.lrn(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        pooled_conv2 = tf.nn.max_pool2d(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        ## Third convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 128],stddev=1e-1), name="conv3_weights") 
        conv = tf.nn.conv2d(pooled_conv2, kernel, [1, 1, 1, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([128]), name="conv3_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv3 = tf.nn.leaky_relu(conv_with_bias, name="conv3")

        ## Fourth convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 128, 256],stddev=1e-1), name="conv4_weights") 
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([256]), name="conv4_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv4 = tf.nn.leaky_relu(conv_with_bias, name="conv4")

        ## Fifth convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 384],stddev=1e-1), name="conv5_weights")
        conv = tf.nn.conv2d(conv4, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([384]), name="conv5_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv5 = tf.nn.leaky_relu(conv_with_bias, name="conv5")

        ## 6th convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 384, 512],stddev=1e-1), name="conv6_weights")
        conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([512]), name="conv6_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv6 = tf.nn.leaky_relu(conv_with_bias, name="conv6")

        ## 7th convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 768],stddev=1e-1), name="conv7_weights")
        conv = tf.nn.conv2d(conv6, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([768]), name="conv7_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv7 = tf.nn.leaky_relu(conv_with_bias, name="conv7")

        ## 8th convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 768, 768],stddev=1e-1), name="conv8_weights")
        conv = tf.nn.conv2d(conv7, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([768]), name="conv8_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv8 = tf.nn.leaky_relu(conv_with_bias, name="conv8")

        ## 9th convolutional layer
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 768, 768],stddev=1e-1), name="conv8_weights")
        conv = tf.nn.conv2d(conv8, kernel, [1, 2, 2, 1], padding="SAME")
        bias = tf.Variable(tf.random.truncated_normal([768]), name="conv9_bias")
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv9 = tf.nn.leaky_relu(conv_with_bias, name="conv9")

        ## Fully connected layers
        
        conv9 = tf.keras.layers.Flatten()(conv9) # tf.flatten
        # fc_size_in = 768
        fc_size_in = conv9.shape[-1]
        
        # First fully connected layer
        weights = tf.Variable(tf.random.truncated_normal([fc_size_in, fc_size]), name="fc1_weights")
        bias = tf.Variable(tf.random.truncated_normal([fc_size]), name="fc1_bias")
        fc1 = tf.matmul(conv9, weights) + bias
        fc1 = tf.nn.leaky_relu(fc1, name="fc1")
        fc1 = tf.nn.dropout(fc1, rate = (self.dropout))

        # Second fully connected layer
        weights = tf.Variable(tf.random.truncated_normal([fc_size, fc_size]), name="fc2_weights")
        bias = tf.Variable(tf.random.truncated_normal([fc_size]), name="fc2_bias")
        fc2 = tf.matmul(fc1, weights) + bias
        fc2 = tf.nn.leaky_relu(fc2, name="fc2")
        fc2 = tf.nn.dropout(fc2, rate = (self.dropout))

        # Output layer
        weights = tf.Variable(tf.zeros([fc_size, n_classes]), name="output_weight")
        bias = tf.Variable(tf.random.truncated_normal([n_classes]), name="output_bias")
        self.out = tf.matmul(fc2, weights) + bias
