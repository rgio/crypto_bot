from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import datetime
import tempfile
import pathlib
import argparse
import json
import pdb
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


# Local Imports
import hparams as hp
import price_data as pdata
import cnn
import loss_value as lv
import print_results as prnt
import functions as fn
import poloniex_api as pnx


def main():
    hparams = hp.set_hparams()
    params = hp.set_params()
    crypto_bot = CryptoBot(hparams, params)


class CryptoBot:
    def __init__(self, hparams, params, test=False, tuning=False):
        self.hparams = hparams
        self.params = params
        self.test = test
        self.tuning = tuning
        
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
        self.basedir = self.create_basedir()
        self.save_hparams_pickle()
        self.tensorboard_callback()
        self.rest()

    def create_basedir(self):
        if self.tuning:
            self.base = 'tmp/tuning/'
        else:
            self.base = 'tmp/output/'
        basedir = self.base + self.timestamp + '/'
        fn.check_path(basedir)
        return basedir

    def save_hparams_pickle(self):
        # save hparams with model
        pickle_protocol = pickle.HIGHEST_PROTOCOL
        filepath = os.path.join(self.basedir, 'hparams.pickle')
        with open(filepath, 'wb') as handle:
            pickle.dump(self.hparams.values(),handle, protocol=pickle_protocol)

        # add hparams to master dictionary of models
        hparam_dict = self.hparams.values()
        master_entry = {self.timestamp: hparam_dict}
        master_filepath = os.path.join(self.base, 'hparams_master.pickle')
        with open(master_filepath, 'wb') as handle:
            pickle.dump(master_entry, handle, protocol=pickle_protocol)

    def tensorboard_callback(self):
        callback_log = TensorBoard(
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False)
        return callback_log

    def rest(self):
        try:
            if self.test:
                input_array = pdata.read_data('data/test/')
            else:
                input_array = pdata.read_data()
        except:
            if self.test:
                pnx.fetch_data(test=True)
                input_array = pdata.read_data('data/test/')
            else:
                pnx.fetch_data()
                input_array = pdata.read_data()
        total_time_steps = input_array.shape[1]
        train_size = int(total_time_steps*0.7)
        validation_size = int(total_time_steps*0.15)
        test_size = int(total_time_steps*0.15)
        train, validation, test = pdata.split_data(input_array, train_size, validation_size, test_size)
        train_data, train_labels = pdata.get_data(train, self.hparams.window_size, self.hparams.stride)
        print('\nNumber of training data steps = %d' % train_labels.shape[0])
        validation_data, validation_labels = pdata.get_data(validation, self.hparams.window_size, self.hparams.stride)
        validation_labels = np.reshape(validation_labels, (validation_labels.shape[0], self.params.num_coins))
        btc_btc = np.ones( (1, validation_labels.shape[0]), dtype=np.float32)
        validation_labels = np.insert(validation_labels, 0, btc_btc, axis=1)
        validation_path = self.basedir + 'validation/'
        fn.check_path(validation_path)
        opt_val_portfolio, opt_val_port_return = pdata.calc_optimal_portfolio(validation_labels, validation_path)
        test_data, test_labels = pdata.get_data(test, self.hparams.window_size, self.hparams.stride)
        test_labels = np.reshape(test_labels, (test_labels.shape[0], self.params.num_coins))
        btc_btc = np.ones( (1, test_labels.shape[0]), dtype=np.float32)
        test_labels = np.insert(test_labels, 0, btc_btc, axis=1)
        test_path = self.basedir + 'test/'
        fn.check_path(test_path)
        opt_test_portfolio, opt_test_port_return = pdata.calc_optimal_portfolio(test_labels, test_path)

        # Create the model
        input_prices = tf.placeholder(tf.float32, [None, self.params.num_coins, self.hparams.window_size, self.params.num_input_channels])
        labels = tf.placeholder(tf.float32, [None, self.params.num_coins+1])
        init_weights = tf.placeholder(tf.float32, [None, self.params.num_coins+1])
        batch_size = tf.placeholder(tf.int32)

        # Build the graph
        weights, keep_prob = cnn.cnn_model(input_prices, init_weights, self.hparams, self.params)

        # Define the loss
        with tf.name_scope('loss'):
            loss = lv.calc_minus_log_rate_return(labels, weights, init_weights, batch_size)
        loss = tf.reduce_mean(loss)

        # Define the optimizer
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(self.hparams.learning_rate).minimize(loss)

        # Define the accuracy of the model
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(weights, axis=1), tf.argmax(labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define the testing conditions
        with tf.name_scope('value'):
            value = lv.calc_portfolio_value_change(labels, weights, init_weights, batch_size)
        final_value = tf.reduce_prod(value)

        # Decide where the graph and model is stored
        path_to_graph = self.basedir + 'graph/'
        # train_writer.add_graph(tf.get_default_graph())
        saver = tf.train.Saver()
        timestamp = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())
        timestamp_path = self.basedir + '/timestamps'
        fn.check_path(timestamp_path)
        timestamp_file = timestamp_path + '/timestamps.txt'
        with open(timestamp_file, 'w') as f:
            f.write('%s' % timestamp)
        path_to_model_dir = self.basedir + 'model/'
        fn.check_path(path_to_model_dir)
        pathlib.Path(path_to_model_dir).mkdir(parents=True, exist_ok=True)
        prnt.print_hyperparameters(self.hparams, path_to_model_dir)
        path_to_final_model = path_to_model_dir + 'cnn_model.ckpt'
        path_to_best_model = path_to_model_dir + 'cnn_best_model.ckpt'
        best_val_value = 0.0 # used to save

        # Stores our portfolio weights
        memory_array = np.random.rand(train_data.shape[0], self.params.num_coins+1)

        # Run the training and testing
        train_path = self.basedir + 'train/'
        fn.check_path(train_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('\nThe TensorFlow dataflow graph will be saved in %s\n' % path_to_graph)
            train_writer = tf.summary.FileWriter(path_to_graph, sess.graph)
            train_writer.add_graph(tf.get_default_graph())
            batch = pdata.get_next_price_batch(train_data, train_labels, 0, self.hparams, self.params)
            #pdb.set_trace()
            input_weights = weights.eval(feed_dict={input_prices: batch[0], labels: batch[1],
                    init_weights: memory_array[:self.hparams.batch_size], batch_size: self.hparams.batch_size, keep_prob: 1.0})
            memory_array[:self.hparams.batch_size] = input_weights
            for i in range(1, self.hparams.num_training_steps):
                batch = pdata.get_next_price_batch(train_data, train_labels, i, self.hparams, self.params)
                input_weights_batch = pdata.get_specific_price_batch(train_data, train_labels, batch[2]-1, self.hparams, self.params)
                input_weights = weights.eval(feed_dict={input_prices: input_weights_batch[0], labels: input_weights_batch[1],
                    init_weights: memory_array[batch[2]:batch[2]+self.hparams.batch_size], batch_size: self.hparams.batch_size, keep_prob: 1.0})
                memory_array[batch[2]:batch[2]+self.hparams.batch_size] = input_weights
                if i % 1000 == 0:
                    pdata.calc_optimal_portfolio(batch[1], train_path)
                    train_value = final_value.eval(feed_dict={input_prices: batch[0], labels: batch[1],
                        init_weights: input_weights, batch_size: self.hparams.batch_size, keep_prob: 1.0})
                    train_accuracy = accuracy.eval(feed_dict={input_prices: batch[0], labels: batch[1],
                        init_weights: input_weights, batch_size: self.hparams.batch_size, keep_prob: 1.0})
                    print('Step = %d\nBatch = %d\nTrain_accuracy = %g\nTrain_value = %g' % (i, batch[2], train_accuracy, train_value))
                if i % 10000 == 0 or i == self.hparams.num_training_steps-1:
                    validation_weights = np.zeros((1, validation_labels.shape[1]))
                    validation_weights[0,0]  = 1.0
                    v = np.ones((validation_labels.shape[0]))
                    portfolio_value = 1.0
                    for i in range(0, validation_labels.shape[0]):
                        v_labels = np.reshape(validation_labels[i,:], (1, validation_labels.shape[1]))
                        v_data = np.reshape(validation_data[i,:], (1, validation_data.shape[1], validation_data.shape[2], validation_data.shape[3]))
                        v[i] = final_value.eval(feed_dict={input_prices: v_data, labels: v_labels,
                                    init_weights: validation_weights, batch_size: 1, keep_prob: 1.0})
                        portfolio_value = portfolio_value*v[i]
                        validation_weights = weights.eval(feed_dict={input_prices: v_data, labels: v_labels,
                            init_weights: validation_weights, batch_size: 1, keep_prob: 1.0})
                    print('validation_steps %d, validation_time = %g validation_value %g' % (validation_labels.shape[0], validation_labels.shape[0]/48.0, portfolio_value))
                    if (portfolio_value > best_val_value):
                        save_path_best_model = saver.save(sess, path_to_best_model)
                        best_val_value = portfolio_value
                        self.final_value = best_val_value
                        models_dict = self.get_models_dict()
                        models_dict[best_val_value] = save_path_best_model
                        self.set_models_dict(models_dict)
                        np.savetxt(validation_path + 'proper_validation_returns.out', v, fmt='%.8f', delimiter=' ')
                        print('new best validation value, best model weights saved in %s\n' % save_path_best_model)
                    train_step.run(feed_dict={input_prices: batch[0], labels: batch[1],
                    init_weights: input_weights, batch_size: self.hparams.batch_size, keep_prob: self.hparams.dropout_keep_prob})

            # Save the results
            save_path_final_model = saver.save(sess, path_to_final_model)
            print('The final model weights are saved in %s\n' % save_path_final_model)

    def get_value(self):
        return self.final_value

    @staticmethod
    def set_models_dict(d):
        with open('models_dict.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_models_dict():
        try:
            with open('models_dict.pickle', 'rb') as handle:
                return pickle.load(handle)
        except:
            return {}

    @staticmethod
    def get_hparams_dict():
        try:

            with open('models_dict.pickle', 'rb') as handle:
                return models_dict
        except:
            return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory for storing input data')
    parser.add_argument('--hparams', type=str, default=None, help='Comma separated list of "name=value" pairs.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
