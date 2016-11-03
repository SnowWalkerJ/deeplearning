import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    @staticmethod
    def stem_layer(net):
        with tf.name_scope('stem'):
            net = tf_learn.layers.conv2d(net, filter_size=3, depth=32, strides=[1, 2, 2, 1], padding='VALID')
            net = tf_learn.layers.conv2d(net, filter_size=3, depth=32, strides=1, padding='VALID')
            net = tf_learn.layers.conv2d(net, filter_size=3, depth=64, strides=1, padding='SAME')
            left = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            right = tf_learn.layers.conv2d(net, depth=96, filter_size=3, strides=[1, 2, 2, 1], padding='VALID', activation='relu')
            net = tf.concat(3, [left, right])
            left = tf_learn.layers.conv2d(net, filter_size=1, depth=64, strides=1, activation='relu')
            left = tf_learn.layers.conv2d(left, filter_size=3, depth=96, strides=1, padding='VALID', activation='relu')
            right = tf_learn.layers.conv2d(net, filter_size=1, depth=64, strides=1, activation='relu')
            right = tf_learn.layers.conv2d(right, filter_size=[1, 7], depth=64, strides=1, activation='relu')
            right = tf_learn.layers.conv2d(right, filter_size=[7, 1], depth=64, strides=1, activation='relu')
            right = tf_learn.layers.conv2d(right, filter_size=3, depth=96, strides=1, padding='VALID', activation='relu')
            net = tf.concat(3, [left, right])
            left = tf_learn.layers.conv2d(net, filter_size=3, depth=192, padding='VALID', strides=2, activation='relu')
            right = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            net = tf.concat(3, [left, right])
        return net

    @staticmethod
    def inception_a(net, name):
        with tf.name_scope('Inception-A-%s' % name):
            net1 = tf.nn.avg_pool(net, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
            net1 = tf_learn.layers.conv2d(net1, filter_size=1, depth=96, strides=1)
            net2 = tf_learn.layers.conv2d(net, filter_size=1, depth=96, strides=1)
            net3 = tf_learn.layers.conv2d(net, filter_size=1, depth=64, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=3, depth=96, strides=1)
            net4 = tf_learn.layers.conv2d(net, filter_size=1, depth=64, strides=1)
            net4 = tf_learn.layers.conv2d(net4, filter_size=3, depth=96, strides=1)
            net4 = tf_learn.layers.conv2d(net4, filter_size=3, depth=96, strides=1)
            net = tf.concat(3, [net1, net2, net3, net4])
        return net

    @staticmethod
    def reduction_a(net):
        with tf.name_scope("Reduction-A"):
            base_depth = net.get_shape().as_list()[-1]
            net1 = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
            net2 = tf_learn.layers.conv2d(net, filter_size=3, depth=base_depth, strides=2, padding='VALID')
            net3 = tf_learn.layers.conv2d(net, filter_size=1, depth=base_depth, strides=1, padding='SAME')
            net3 = tf_learn.layers.conv2d(net3, filter_size=3, depth=base_depth, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=3, depth=base_depth, strides=2, padding='VALID')
            net = tf.concat(3, [net1, net2, net3])
        return net

    @staticmethod
    def inception_b(net, name):
        with tf.name_scope("Inception-B-%s" % name):
            net1 = tf.nn.avg_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
            net1 = tf_learn.layers.conv2d(net1, filter_size=1, depth=128, strides=1, padding='SAME')
            net2 = tf_learn.layers.conv2d(net, filter_size=1, depth=384, strides=1)
            net3 = tf_learn.layers.conv2d(net, filter_size=1, depth=192, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=[1, 7], depth=224, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=[7, 1], depth=256, strides=1)
            net4 = tf_learn.layers.conv2d(net, filter_size=1, depth=192, strides=1) 
            net4 = tf_learn.layers.conv2d(net4, filter_size=[1, 7], depth=192, strides=1)
            net4 =  tf_learn.layers.conv2d(net4, filter_size=[7, 1], depth=224, strides=1) 
            net4 = tf_learn.layers.conv2d(net4, filter_size=[1, 7], depth=224, strides=1) 
            net4 = tf_learn.layers.conv2d(net4, filter_size=[1, 7], depth=256, strides=1)
            net = tf.concat(3, [net1, net2, net3, net4])
        return net

    @staticmethod
    def reduction_b(net):
        with tf.name_scope("Reduction-B"):
            net1 = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
            net2 = tf_learn.layers.conv2d(net, filter_size=1, depth=192, strides=1)
            net2 = tf_learn.layers.conv2d(net2, filter_size=3, depth=192, strides=2, padding='VALID')
            net3 = tf_learn.layers.conv2d(net, filter_size=1, depth=256, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=[1, 7], depth=256, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=[7, 1], depth=320, strides=1)
            net3 = tf_learn.layers.conv2d(net3, filter_size=3, depth=320, strides=2, padding='VALID')
            net = tf.concat(3, [net1, net2, net3])
        return net

    @staticmethod
    def inception_c(net, name):
        with tf.name_scope('Inception-C-%s' % name):
            net1 = tf.nn.avg_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
            net1 = tf_learn.layers.conv2d(net1, filter_size=1, depth=256, strides=1)
            net2 = tf_learn.layers.conv2d(net, filter_size=1, depth=256, strides=1)
            net3 = tf_learn.layers.conv2d(net, filter_size=1, depth=384, strides=1)
            net31 = tf_learn.layers.conv2d(net3, filter_size=[1, 3], depth=256, strides=1)
            net32 = tf_learn.layers.conv2d(net3, filter_size=[3, 1], depth=256, strides=1)
            net4 = tf_learn.layers.conv2d(net, filter_size=1, depth=384, strides=1)
            net4 = tf_learn.layers.conv2d(net4, filter_size=[1, 3], depth=448, strides=1)
            net4 = tf_learn.layers.conv2d(net4, filter_size=[3, 1], depth=512, strides=1)
            net41 = tf_learn.layers.conv2d(net4, filter_size=[3, 1], depth=256, strides=1)
            net42 = tf_learn.layers.conv2d(net4, filter_size=[1, 3], depth=256, strides=1)
            net = tf.concat(3, [net1, net2, net31, net32, net41, net42])
        return net

    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.7,
                'evaluate': 1.0,
            },
            'lr': 0.01
        }
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        lr = self.register_placeholder('lr', shape=None, dtype=tf.float32)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        net = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128.0
        net = self.stem_layer(net)
        with tf.name_scope("Inception-A"):
            for i in xrange(4):
                net = self.inception_a(net, str(i))

        net = self.reduction_a(net)

        with tf.name_scope("Inception-B"):
            for i in xrange(7):
                net = self.inception_b(net, str(i))

        net = self.reduction_b(net)

        with tf.name_scope('Inception-C'):
            for i in xrange(3):
                net = self.inception_c(net, str(i))

        base_size = net.get_shape().as_list()[1]
        net = tf.nn.avg_pool(net, [1, base_size, base_size, 1], [1, base_size, base_size, 1], 'VALID', name='AvgPool')
        net = tf.nn.dropout(net, keep_prob, name='dropout')
        self.output_tensor = tf_learn.layers.fully_connection(net, 2, activation='linear', name='output_tensor')
        
        self.target_tensor = tf.placeholder(tf.int32, [None], name='target_tensor')
        with tf.name_scope('one_hot'):
            self.one_hot_labels = tf.one_hot(self.target_tensor, 2, name='one_hot_labels')
        with tf.name_scope('loss'):
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_tensor, self.one_hot_labels, name='cross_entropy'))
            self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.nn.softmax(self.output_tensor)) * self.one_hot_labels, reduction_indices=[1]))
        # self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.train_op = tf.train.RMSPropOptimizer(lr, momentum=0.9).minimize(self.loss)
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

    def on_train_finish_epoch(self):
        if self.epoch % 3 == 0:
            self.placeholders['lr'] *= 0.96
        self.run_summary(self.epoch + 1)

    def on_before_train(self):
	    self.run_summary(0)



