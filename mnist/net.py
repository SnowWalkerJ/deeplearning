import tensorflow as tf
import tflearn
from time import sleep
import random


class Model(object):
    def __init__(self, **kwargs):
        self.train_op = kwargs['train_op']
        self.x = kwargs['x']
        self.y = kwargs['y']
        self.y_hat = kwargs['y_hat']
        self.graph = kwargs['graph']
        self.evaluation = kwargs.get('evaluation')
        self.params = kwargs.get('params') or {}
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.initialize_all_variables())

    def train(self, x_data, y_data, valid_x=None, valid_y=None, batch_size=100, n_epoch=10, params=None, eval_params=None, nsleep=0, shuffle=True):
        _params = {}
        for key in params:
            if key in self.params:
                _params[self.params[key]] = params[key]
        _eval_params = {}
        _eval_params.update(_params)
        for key in eval_params:
            if key in self.params:
                _eval_params[self.params[key]] = eval_params[key]
        n_records = len(x_data)
        with self.session.as_default():
            for epoch in xrange(n_epoch):
                index = range(n_records)
                if shuffle:
                    random.shuffle(index)
                for batch_no in xrange(0, n_records, batch_size):
                    upper_limit = min(batch_no + batch_size, n_records)
                    batch_x, batch_y = x_data[index[batch_no:upper_limit]], y_data[index[batch_no:upper_limit]]
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    feed_dict.update(_params)
                    self.train_op.run(feed_dict)
                if self.evaluation and valid_x is not None and valid_y is not None:
                    # TODO: add evaluation
                    valid_feed_dict = {self.x: valid_x, self.y: valid_y}
                    valid_feed_dict.update(_eval_params)
                    result = dict(zip(self.evaluation.keys(), self.session.run(self.evaluation.values(), valid_feed_dict)))
                    self.print_evaluation(epoch, n_epoch, result)
                else:
                    self.print_evaluation(epoch, n_epoch)
                self.saver.save(self.session, 'models/', epoch)
                sleep(nsleep)

        self.saver.save(self.session, 'models/model1')

    def load(self, filename):
        self.saver.restore(self.session, filename)

    def predict(self, x_data, **kwargs):
        feed_dict = {self.x: x_data}
        for key in kwargs:
            if key in self.params:
                feed_dict[self.params[key]] = kwargs[key]
        with self.session.as_default():
            return self.y_hat.eval(feed_dict)

    @staticmethod
    def print_evaluation(epoch, n_epoch, result=None):
        print "epoch: %d / %d completed" % (epoch+1, n_epoch)
        if result:
            for k, v in result.iteritems():
                print "%s: %f" % (k, v)
        print "#" * 10


def build_net():
    graph = tf.Graph()
    with graph.as_default():
        keep_prob = tf.placeholder(tf.float32)
        x = tflearn.input_data((None, 28, 28, 1), name='input')
        net = tflearn.layers.conv_2d(x, 16, 3, activation='relu', regularizer='L2')
        net = tflearn.layers.local_response_normalization(net)
        net = tflearn.layers.max_pool_2d(net, 3)
        net = tflearn.layers.conv_2d(net, 64, 3, activation='relu', regularizer='L2')
        net = tflearn.layers.local_response_normalization(net)
        net = tflearn.layers.max_pool_2d(net, 3)
        net = tflearn.layers.fully_connected(net, 512, activation='tanh', regularizer='L1')
        net = tflearn.layers.dropout(net, keep_prob)
        net = tflearn.layers.fully_connected(net, 2048, activation='tanh')
        net = tflearn.layers.dropout(net, keep_prob)
        y_hat = tflearn.fully_connected(net, 10, activation='linear', name='output')
        y = tf.placeholder(tf.float32, (None, 10), name='target')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y))
        train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return Model(train_op=train_op, x=x, y=y, y_hat=y_hat, graph=graph, evaluation={'acc': accuracy}, params=dict(keep_prob=keep_prob))
