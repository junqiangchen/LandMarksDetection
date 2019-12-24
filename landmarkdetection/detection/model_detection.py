'''

'''
import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .layer import (conv2d, deconv2d, normalizationlayer2d, crop_and_concat2d, resnet_Add,
                    weight_xavier_init, bias_variable, save_images)
from .utils import normalize, resize_image_itk, reduce_dimension


def conv_bn_relu_drop(x, kernal, phase, drop, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv2d(x, W) + B
        conv = normalizationlayer2d(conv, is_train=phase, height=height, width=width, norm_type='group',
                                    scope=scope)
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W, 2) + B
        conv = normalizationlayer2d(conv, is_train=phase, height=height, width=width, norm_type='group',
                                    scope=scope)
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv2d(x, W, samefeture) + B
        conv = tf.nn.relu(conv)
        return conv


def conv_sigmod(x, kernal, scope=None, activeflag=True):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W) + B
        if activeflag:
            conv = tf.nn.sigmoid(conv)
        return conv


def _create_conv_net(X, image_width, image_height, image_channel, phase, drop, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, image_channel, 64), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 64, 64), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 64, 128), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 128, 128), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 128, 128), samefeture=True, scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat2d(layer4, deconv1)
    _, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 256, 128), height=H, width=W, phase=phase, drop=drop,
                               scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 128, 128), height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 64, 128), samefeture=False, scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat2d(layer3, deconv2)
    _, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat2d(layer2, deconv3)
    _, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 64, 64), samefeture=True, scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat2d(layer1, deconv4)
    _, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 128, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 64, 64), height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map_logit = conv_sigmod(x=layer9, kernal=(1, 1, 64, n_class), scope='output', activeflag=False)
    return output_map_logit


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class Vnet2dlandmarkdetectionModule(object):
    """
        A Vnet2dlandmarkdetectionModule implementation,make sure all landmarks in the input image
        :param image_height: number of height in the input image
        :param image_width: number of width in the input image
        :param image_depth: number of depth in the input image
        :param channels: number of channels in the input image
        :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_height, image_width, channels=1, numclass=1, costname=("L2-loss",),
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.numclass = numclass
        self.labelchannels = numclass
        self.dimension = 2

        self.X = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.numclass])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')

        self.Y_pred_logit = _create_conv_net(self.X, self.image_width, self.image_height, self.channels, self.phase,
                                             self.drop, self.numclass)
        self.cost = self.__get_cost(self.Y_pred_logit, self.Y_gt, costname[0])

        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_cost(self, Y_pred, Y_gt, cost_name):
        if cost_name == "L2-loss":
            loss = tf.nn.l2_loss(Y_pred - Y_gt)
            return loss
        if cost_name == "mse":
            loss = tf.losses.mean_squared_error(Y_gt, Y_pred)
            return loss

    def __get_landmark(self, image):
        max_index = np.argmax(image)
        coord = np.array(np.unravel_index(max_index, dims=image.shape), np.int)
        value = image[tuple(coord)]
        return coord, value

    def __get_landmarks(self, predictiamge):
        coords = []
        values = []
        for image in np.rollaxis(predictiamge, axis=self.dimension):
            coord, value = self.__get_landmark(image)
            coords.append(coord)
            values.append(value)
        coords_array = np.array(coords)
        values_array = np.array(values)
        return coords_array, values_array

    def __get_metric(self, Y_pred, Y_gt, metric_name="EuclideanDistance"):
        num_samples = Y_gt.shape[0]
        mertic = 0
        if metric_name == "EuclideanDistance":
            for num in range(num_samples):
                Y1 = Y_pred[num]
                Y2 = Y_gt[num]
                pd_coords, pd_values = self.__get_landmarks(Y1)
                gt_coords, gt_values = self.__get_landmarks(Y2)
                distance = pd_coords - gt_coords
                distance_vector = distance.flatten()
                mertic = mertic + np.linalg.norm(distance_vector)
            mertic = mertic / num_samples
            return mertic

    def __loadnumtraindata(self, train_images, train_lanbels, num_sample, num_sample_index_in_epoch):
        """
        load train data
        :param train_images:
        :param train_lanbels:
        :param num_sample:
        :param num_sample_index_in_epoch:
        :return:
        """
        subbatch_xs = np.empty((num_sample, self.image_height, self.image_width, self.channels))
        subbatch_ys = np.empty((num_sample, self.image_height, self.image_width, self.labelchannels))
        batch_xs_path, batch_ys_path, num_sample_index_in_epoch = _next_batch(train_images, train_lanbels,
                                                                              num_sample, num_sample_index_in_epoch)
        for num in range(len(batch_xs_path)):
            image = np.load(batch_xs_path[num])
            label = np.load(batch_ys_path[num])
            subbatch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_width, self.channels))
            subbatch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_width, self.labelchannels))
        subbatch_xs = subbatch_xs.astype(np.float)
        subbatch_ys = subbatch_ys.astype(np.float)
        return subbatch_xs, subbatch_ys, num_sample_index_in_epoch

    def train(self, train_images, train_labels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1, showwind=[8, 8]):
        num_sample = 100
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model\\"):
            os.makedirs(logs_path + "model\\")
        model_path = logs_path + "model\\" + model_path
        train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        if os.path.exists(model_path):
            saver.restore(sess, model_path)

        DISPLAY_STEP = 1
        index_in_epoch = 0
        num_sample_index_in_epoch = 0

        train_epochs = train_images.shape[0] * train_epochs
        for i in range(train_epochs):
            # Extracting num_sample images and labels from given data
            if i % num_sample == 0 or i == 0:
                subbatch_xs, subbatch_ys, num_sample_index_in_epoch = self.__loadnumtraindata(train_images,
                                                                                              train_labels, num_sample,
                                                                                              num_sample_index_in_epoch)
            # get new batch
            batch_xs, batch_ys, index_in_epoch = _next_batch(subbatch_xs, subbatch_ys, batch_size, index_in_epoch)
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss = sess.run(self.cost, feed_dict={self.X: batch_xs,
                                                            self.Y_gt: batch_ys,
                                                            self.lr: learning_rate,
                                                            self.phase: 1,
                                                            self.drop: dropout_conv})
                pred = sess.run(self.Y_pred_logit, feed_dict={self.X: batch_xs,
                                                              self.Y_gt: batch_ys,
                                                              self.phase: 1,
                                                              self.drop: 1})
                train_accuracy = self.__get_metric(pred, batch_ys)
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))
                gt = np.reshape(batch_ys[0], (self.image_height, self.image_width, self.labelchannels))
                gt = gt.astype(np.float)
                save_images(gt, showwind, path=logs_path + 'gt_%d_epoch.png' % (i))
                result = np.reshape(pred[0], (self.image_height, self.image_width, self.labelchannels))
                result = result.astype(np.float)
                save_images(result, showwind, path=logs_path + 'predict_%d_epoch.png' % (i))
                save_path = saver.save(sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10
                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()
        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        assert self.image_width == test_images.shape[0], \
            'prediction process the input size is not equal vnet input size'
        test_images = np.reshape(test_images, (self.image_height, self.image_width, self.channels))
        y_dummy = np.zeros(shape=(self.image_height, self.image_width, self.labelchannels))
        test_images = test_images.astype(np.float)
        pred = self.sess.run(self.Y_pred_logit, feed_dict={self.X: [test_images],
                                                           self.Y_gt: [y_dummy],
                                                           self.phase: 1,
                                                           self.drop: 1})
        result = pred.astype(np.float)
        result = np.reshape(result, (self.image_height, self.image_width, self.labelchannels))
        return result

    def inference(self, filepath):
        # 1 load image with itk
        src_itkimage = sitk.ReadImage(filepath, sitk.sitkFloat32)
        # 2 reduce dimension image with itk
        src_itkimage = reduce_dimension(src_itkimage)
        # 3 resize to vnet size and predict
        itkimagesize = src_itkimage.shape
        offsetx, offsety = self.image_width / itkimagesize[0], self.image_height / itkimagesize[1]
        rezieitkimage = resize_image_itk(src_itkimage, newSize=(self.image_width, self.image_height))
        input_array = sitk.GetArrayFromImage(rezieitkimage)
        input_array = np.transpose(input_array, (1, 0))
        input_array = normalize(input_array)  # normalize image to mean 0 std 1
        heatmaps_array = self.prediction(input_array)
        # 4 get landmark coords
        coords_array, values_array = self.__get_landmarks(heatmaps_array)
        # 5 resize landmarks_coords to src image size
        resize_coords_array = np.around(coords_array * np.array((1 / offsetx, 1 / offsety)))
        return resize_coords_array, values_array
