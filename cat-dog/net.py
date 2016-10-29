import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):

    @staticmethod
    def inception_layer(input_tensor, name, large=False):
        base_depth = input_tensor.get_shape().as_list()[-1]
        with tf.name_scope(name):
            conv_left = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=1, strides=1, activation='relu', name='1x1x%d' % base_depth)
            conv_right = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=1, strides=1, activation='relu', name='1x1x%d' % base_depth)
            conv_right = tf_learn.layers.conv2d(conv_right, depth=base_depth, filter_size=3, strides=1, activation='relu', name='3x3x%d' % base_depth)
            stacks = [conv_left, conv_right]
            if large:
               conv_large = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=2, strides=1, activation='relu', name='1x1x%d' % base_depth)           
               conv_large = tf_learn.layers.conv2d(conv_large, depth=base_depth, filter_size=[5, 1], strides=1, activation='relu', name='5x1x%d' % base_depth)
               conv_large = tf_learn.layers.conv2d(conv_large, depth=base_depth, filter_size=[1, 5], strides=1, activation='relu', name='1x5x%d' % base_depth)
               stacks.append(conv_large)
            output_tensor = tf.concat(3, stacks)
            output_tensor = tf_learn.layers.conv2d(output_tensor, depth=int(output_tensor.get_shape().as_list()[-1]/1.5), filter_size=1, strides=1, activation='relu', name='1x1x%d' % int(base_depth*2/1.5)) 
        return output_tensor
    
    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.5,
                'evaluate': 1.0,
            },
        }
        self.global_step = tf.Variable(0, trainable=False)
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        lr = tf.train.exponential_decay(0.01, self.global_step, 1500, 0.96, staircase=True)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        with tf.name_scope('normalization'):
            cast_float = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128.0
           
        with tf.name_scope('conv0'):
            net = tf_learn.layers.conv2d(cast_float, depth=32, filter_size=3, strides=[1, 2, 2, 1], activation='relu', name='3x3x32', padding='VALID')
            net = tf_learn.layers.conv2d(net, depth=32, filter_size=1, strides=1, activation='relu', name='1x1x32', padding='VALID')
            net = tf_learn.layers.conv2d(net, depth=64, filter_size=3, strides=1, activation='relu', name='3x3x64', padding='VALID')
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            net = tf_learn.layers.conv2d(net, depth=64, filter_size=1, strides=1, activation='relu', name='1x1x64')
            net = tf_learn.layers.conv2d(net, depth=64, filter_size=3, strides=1, activation='relu', name='3x3x64')

        net = self.inception_layer(net, 'layer1')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        net = self.inception_layer(net, 'layer2')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        net = self.inception_layer(net, 'layer3')
        net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        net = self.inception_layer(net, 'layer4')
        
        image_size = net.get_shape().as_list()[1]
        image_depth = net.get_shape().as_list()[-1]
        net = tf.nn.max_pool(net, [1, image_size, image_size, 1], [1, image_size, image_size, 1], 'VALID')
        net = tf.reshape(net, [-1, image_depth])
        self.output_tensor = tf_learn.layers.fully_connection(net, 2, activation='softmax', name='output_tensor')
        self.target_tensor = tf.placeholder(tf.int32, [None], name='target_tensor')
        with tf.name_scope('one_hot'):
            self.one_hot_labels = tf.one_hot(self.target_tensor, 2, name='one_hot_labels')
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(self.output_tensor) * self.one_hot_labels, reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(tf.equal(self.target_tensor,
                                                  tf.cast(tf.argmax(self.output_tensor, 1), tf.int32)),
                                         tf.float32),
                                 name='accuracy')
        self.evaluation_dict = {
            'loss': self.loss,
            'acc': acc,
        }
        tf.scalar_summary('accuracy', acc)
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()

    def on_train_finish_batch(self):
        global_step = self.sess.run(tf.assign_add(self.global_step, 1))
        
        if global_step % 70 == 0:
            self.run_summary(global_step / 10)


    def on_before_train(self):
        self.run_summary(0)



