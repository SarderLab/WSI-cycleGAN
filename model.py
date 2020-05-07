from __future__ import division
import os
import time
import random
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dirA = '../GTEx_kidney_cortex/'
        self.dataset_dirB = '/hdd/WSI_data/all_svs/human/H&E_from_Kuang/'
        self.dataset_dir = './'

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def input_setup(self):

        '''
        This function sets up the input data pipeline
        takes input at tfrecord files
        '''

        #######################################################################

        # data A input pipeline
        self.data_A_paths = list(glob('{}*.svs'.format(self.dataset_dirA)))
        self.data_A_paths = [str(path) for path in self.data_A_paths]
        random.shuffle(self.data_A_paths)
        image_count = len(self.data_A_paths)
        print('found: {} images from dataset A'.format(image_count))

        self.path_ds_A = tf.data.Dataset.from_tensor_slices(self.data_A_paths)
        self.path_ds_A = self.path_ds_A.repeat(100)
        self.dataset_A = self.path_ds_A.map(lambda filename: tf.py_function(get_random_wsi_region, [filename], tf.float32), num_parallel_calls=40)
        self.dataset_A = self.dataset_A.shuffle(100)
        self.dataset_A = self.dataset_A.batch(batch_size=self.batch_size, drop_remainder=True)
        self.dataset_A = self.dataset_A.prefetch(buffer_size=1)
        self.iterator_A = tf.data.Iterator.from_structure(self.dataset_A.output_types, self.dataset_A.output_shapes)
        self.training_init_op_A = self.iterator_A.make_initializer(self.dataset_A)
        self.get_input_A = self.iterator_A.get_next()

        #######################################################################

        # data B input pipeline
        self.data_B_paths = list(glob('{}*.svs'.format(self.dataset_dirB)))
        self.data_B_paths = [str(path) for path in self.data_B_paths]
        random.shuffle(self.data_B_paths)
        image_count = len(self.data_B_paths)
        print('found: {} images from dataset B'.format(image_count))

        self.path_ds_B = tf.data.Dataset.from_tensor_slices(self.data_B_paths)
        self.path_ds_B = self.path_ds_B.repeat()
        self.dataset_B = self.path_ds_B.map(lambda filename: tf.py_function(get_random_wsi_region, [filename], tf.float32), num_parallel_calls=40)
        self.dataset_B = self.dataset_B.shuffle(100)
        self.dataset_B = self.dataset_B.batch(batch_size=self.batch_size, drop_remainder=True)
        self.dataset_B = self.dataset_B.prefetch(buffer_size=1)
        self.iterator_B = tf.data.Iterator.from_structure(self.dataset_B.output_types, self.dataset_B.output_shapes)
        self.training_init_op_B = self.iterator_B.make_initializer(self.dataset_B)
        self.get_input_B = self.iterator_B.get_next()

    def _build_model(self):
        #Build the data pipeline
        self.input_setup()

        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            self.sess.run([self.training_init_op_A, self.training_init_op_B])

            '''
            dataA = glob('{}/*.svs'.format(self.dataset_dirA))
            dataB = glob('{}/*.svs'.format(self.dataset_dirB))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            '''
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            idx = 0
            while True:
                idx += 1
                try:
                    A_input = self.sess.run(self.get_input_A)
                    B_input = self.sess.run(self.get_input_B)
                    batch_images = np.concatenate([A_input,B_input],axis=-1)

                    '''
                    for idx in range(0, batch_idxs):
                    batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                           dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                    batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                    batch_images = np.array(batch_images).astype(np.float32)
                    '''

                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_str = self.sess.run(
                        [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                        feed_dict={self.real_data: batch_images, self.lr: lr})
                    self.writer.add_summary(summary_str, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])

                    # Update D network
                    _, summary_str = self.sess.run(
                        [self.d_optim, self.d_sum],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.lr: lr})
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print(("Epoch: [%2d] [%4d] time: %4.4f" % (
                        epoch, idx, time.time() - start_time)))

                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(args.sample_dir, epoch, idx)

                    if np.mod(counter, args.save_freq) == 2:
                        self.save(args.checkpoint_dir, counter)

                except tf.errors.OutOfRangeError: # end of epoch
                    print('\nDone with epoch {}'.format(epoch))
                    break

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('{}/*.svs'.format(self.dataset_dirA))
        dataB = glob('{}/*.svs'.format(self.dataset_dirB))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)
        real_A = sample_images[:, :, :, :self.input_c_dim]
        print(real_A.shape)
        real_B = sample_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        print(fake_A.shape)
        save_images(np.concatenate([real_B,fake_A], axis=1), [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(np.concatenate([real_A,fake_B], axis=1), [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
