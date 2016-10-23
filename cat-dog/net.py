import tensorflow as tf
import tf_learn as learn
import tf_learn.models, tf_learn.models.dnn, tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.5,
                'evaluate': 1.0,
            },
            'lr': 1e-1
        }
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        lr = self.register_placeholder('lr', shape=None, dtype=tf.float32)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        cast_float = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128
        with tf.name_scope('layer1'):
            conv1 = tf_learn.layers.conv2d(cast_float, depth=8, filter_size=3, strides=1, activation='relu')
            lrn1 = tf.nn.local_response_normalization(conv1, name='local_response_normalization1')
            stack1 = tf.concat(3, [cast_float, lrn1], name='stack1')
            pool1 = tf.nn.max_pool(stack1, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        with tf.name_scope('layer2'):
            conv2 = tf_learn.layers.conv2d(pool1, depth=32, filter_size=3, strides=1, activation='relu')
            lrn2 = tf.nn.local_response_normalization(conv2, name='local_response_normalization2')
            stack2 = tf.concat(3, [pool1, lrn2], name='stack2')
            pool2 = tf.nn.max_pool(stack2, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        with tf.name_scope('layer3'):
            conv3 = tf_learn.layers.conv2d(pool2, depth=128, filter_size=3, strides=1, activation='relu')
            lrn3 = tf.nn.local_response_normalization(conv3, name='local_response_normalization3')
            stack3 = tf.concat(3, [pool2, lrn3], name='stack3')
            pool3 = tf.nn.max_pool(stack3, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        """
        with tf.name_scope('layer4'):
            conv4 = tf_learn.layers.conv2d(pool3, depth=512, filter_size=3, strides=1, activation='relu')
            lrn4 = tf.nn.local_response_normalization(conv4, name='local_response_normalization4')
            stack4 = tf.concat(3, [pool3, lrn4], name='stack4')
            pool4 = tf.nn.max_pool(stack4, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        """
        flattened = tf_learn.layers.flatten(pool3, name='flatten')
        fc1 = tf_learn.layers.fully_connection(flattened, 1024 * 2, activation='tanh', name='fc1')
        dropped1 = tf.nn.dropout(fc1, keep_prob, name='dropout1')
        fc2 = tf_learn.layers.fully_connection(dropped1, 1024 * 8, activation='tanh', name='fc2')
        dropped2 = tf.nn.dropout(fc2, keep_prob, name='dropout2')
        self.output_tensor = tf_learn.layers.fully_connection(dropped2, 2, activation='linear', name='output_tensor')
        self.target_tensor = tf.placeholder(tf.int32, [None], name='target_tensor')
        self.one_hot_labels = tf.one_hot(self.target_tensor, 2, name='one_hot_labels')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_tensor, self.one_hot_labels, name='cross_entropy'))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(self.target_tensor, tf.cast(tf.argmax(self.output_tensor, 1), tf.int32)), tf.float32), name='accuracy')
        self.evaluation_dict = {
            'loss': self.loss,
            'acc': acc,
        }
        tf.scalar_summary('accuracy', acc)
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('learning rate', lr)
        tf.histogram_summary('fc1_weight', fc1.W)
        self.summary = tf.merge_all_summaries()

    def on_train_finish_epoch(self):
        if self.placeholders['lr'] < 1e-4:
            self.placeholders['lr'] *= 0.9
        else:
            self.placeholders['lr'] *= 0.7





