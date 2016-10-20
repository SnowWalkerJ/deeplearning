import tflearn
import tensorflow as tf


class Model(object):
    def __init__(self, input_tensor, target_tensor, output_tensor, loss_tensor, graph, placeholder=None, evaluation=None):
        self.input_tensor = input_tensor
        self.target_
        self.output_tensor = output_tensor
        self.loss_tensor = loss_tensor
        self.graph = graph
        self.evaluation = evaluation or {}
        self.placeholder = placeholder or {}
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(tf.initialize_all_variables())

   def train(self, x, y, valid_x=None, valid_y=None, batch_size=100, n_epoch=10, shuffle=True, params=None, valid_params=None):
   

