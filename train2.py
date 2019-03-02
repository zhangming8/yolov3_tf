# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import logging
import time

from model.yolo_v3 import YOLO_V3
import config as cfg
from data import Data


class YoloTrain(object):
    def __init__(self):
        self.__frozen = cfg.FROZEN
        self.__classes = cfg.CLASSES
        self.__learn_rate_init = cfg.LEARN_RATE_INIT
        self.__max_epochs = cfg.MAX_PERIODS
        self.__lr_decay_epoch = cfg.lr_decay_epoch
        self.__weights_dir = cfg.WEIGHTS_DIR
        self.__weights_file = cfg.WEIGHTS_FILE
        self.__time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'train', self.__time)
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__save_iter = cfg.SAVE_ITER
        self.__train_data = Data('train')
        self.__test_data = Data('test')

        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.__label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.__label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.__sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.__mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.__lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.__training = True

        self.__yolo = YOLO_V3(self.__training)
        self.__conv_sbbox, self.__conv_mbbox, self.__conv_lbbox, \
        self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox = self.__yolo.build_nework(self.__input_data)
        self.__net_var = tf.global_variables()
        logging.info('Load weights:')
        for var in self.__net_var:
            logging.info(var.op.name)

        self.__loss = self.__yolo.loss(self.__conv_sbbox, self.__conv_mbbox, self.__conv_lbbox,
                                       self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox,
                                       self.__label_sbbox, self.__label_mbbox, self.__label_lbbox,
                                       self.__sbboxes, self.__mbboxes, self.__lbboxes)

        with tf.name_scope('learn'):
            self.__learn_rate = tf.Variable(self.__learn_rate_init, trainable=False, name='learn_rate_init')
            moving_ave = tf.train.ExponentialMovingAverage(self.__moving_ave_decay).apply(tf.trainable_variables())

            self.__trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.__trainable_var_list.append(var)
            optimize0 = tf.train.AdamOptimizer(self.__learn_rate).\
                minimize(self.__loss, var_list=self.__trainable_var_list)
            with tf.control_dependencies([optimize0]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op_with_frozen_variables = tf.no_op()

            optimize1 = tf.train.AdamOptimizer(self.__learn_rate).\
                minimize(self.__loss, var_list=tf.trainable_variables())
            with tf.control_dependencies([optimize1]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op_with_all_variables = tf.no_op()
            if self.__frozen:
                self.__train_op = self.__train_op_with_frozen_variables
                logging.info('freeze the weight of darknet')
                print('freeze the weight of darknet')
            else:
                self.__train_op = self.__train_op_with_all_variables
                logging.info("train all variables")
                print("train all variables")
            for var in self.__trainable_var_list:
                logging.info("++++++ trainable variables list: ++++++" + "\n")
                logging.info('\t' + str(var.op.name).ljust(50) + str(var.shape))

        with tf.name_scope('load_save'):
            self.__load = tf.train.Saver(self.__net_var)
            self.__save = tf.train.Saver(tf.global_variables(), max_to_keep=50)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.__loss)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())

        self.__sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def train(self):
        self.__sess.run(tf.global_variables_initializer())
        ckpt_path = self.__weights_file
        logging.info('Restoring weights from:\t %s' % ckpt_path)
        print('Restoring weights from:\t %s' % ckpt_path)
        self.__load.restore(self.__sess, ckpt_path)
        display_step = 50

        for epoch in range(self.__max_epochs):
            if epoch % self.__lr_decay_epoch == 0:
                learning_rate_value = self.__sess.run(tf.assign(self.__learn_rate, self.__sess.run(self.__learn_rate)/10.0))
                logging.info("change lr: {}".format(learning_rate_value))
                print("change lr: {}".format(learning_rate_value))
            total_train_loss = 0.0
            for step, (batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes) in enumerate(self.__train_data):
                _, summary_value, loss_value = self.__sess.run([self.__train_op, self.__summary_op, self.__loss], feed_dict={
                    self.__input_data: batch_image,
                    self.__label_sbbox: batch_label_sbbox,
                    self.__label_mbbox: batch_label_mbbox,
                    self.__label_lbbox: batch_label_lbbox,
                    self.__sbboxes: batch_sbboxes,
                    self.__mbboxes: batch_mbboxes,
                    self.__lbboxes: batch_lbboxes})
                if np.isnan(loss_value):
                    raise ArithmeticError('The gradient is exploded')
                total_train_loss += loss_value
                if step % display_step or step == 0:
                    continue
                train_loss = total_train_loss / display_step
                total_train_loss = 0.0
                self.__summary_writer.add_summary(summary_value, epoch * len(self.__train_data) + step)
                logging.info('epoch: {}/{}, step: {}/{}, lr: {}, train loss: {}'.format(epoch, self.__max_epochs, step,len(self.__train_data), learning_rate_value, train_loss))
                print('epoch: {}/{}, step: {}/{}, lr: {}, train loss: {}'.format(epoch, self.__max_epochs, step,len(self.__train_data), learning_rate_value, train_loss))

            if (epoch + 1) % self.__save_iter:
                continue

            total_test_loss = 0.0
            for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes in self.__test_data:
                loss_value = self.__sess.run(self.__loss, feed_dict={
                    self.__input_data: batch_image,
                    self.__label_sbbox: batch_label_sbbox,
                    self.__label_mbbox: batch_label_mbbox,
                    self.__label_lbbox: batch_label_lbbox,
                    self.__sbboxes: batch_sbboxes,
                    self.__mbboxes: batch_mbboxes,
                    self.__lbboxes: batch_lbboxes})
                total_test_loss += loss_value
            test_loss = total_test_loss / len(self.__test_data)
            logging.info('epoch: {}/{}, test loss: {}'.format(epoch, self.__max_epochs, test_loss))
            print('epoch: {}/{}, test loss: {}'.format(epoch, self.__max_epochs, test_loss))
            saved_model_name = os.path.join(self.__weights_dir, 'yolo.ckpt-epoch%d-loss%.4f' % (epoch, test_loss))
            self.__save.save(self.__sess, saved_model_name)
            logging.info('Saved model:\t%s' % saved_model_name)
            print('Saved model:\t%s' % saved_model_name)
        self.__summary_writer.close()


if __name__ == '__main__':
    log_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logging.basicConfig(filename='log/train/' + log_time + '.log', format='%(filename)s %(asctime)s\t%(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    logging.info('Batch size for step1 is:\t%d' % cfg.BATCH_SIZE)
    logging.info('Batch size for step2 is:\t%d' % cfg.BATCH_SIZE_STEP2)
    logging.info('Initial learn rate is:\t%f' % cfg.LEARN_RATE_INIT)
    YoloTrain().train()

