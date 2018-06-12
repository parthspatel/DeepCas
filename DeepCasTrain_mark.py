import numpy as np
import csv

import tensorflow as tf

from processing_haeussler import Processing
from DeepCas import DeepCas


def main():

    # Model Paramaters
    #-------------------------------------------------------------------------
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config=config

    session = tf.Session()
    restore = False

    data_shape = [4, 23, 2]
    batch_size = 1500
    val_size = 5000
    epochs = 10000000
    learning_rate = 0.0001  # not used in DeepCas.py, using adaptive learning rate

    sub_sample = None

    use_batch_norm = True
    use_dropout = False

    tensorboard_directory = './tmp/tensorboard/1_2x3x3_up_sample_batch_norm_3'

    # Conv2d inputs
    #     filters : Integer, dimensionality of the output space (ie. the number of filters in the convolution)
    #     kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    #                   Can be a single integer to specify the same value for all spatial dimensions
    #     strides : An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    #               Can be a single integer to specify the same value for all spatial dimensions
    conv2d_specifications = [[{'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 160, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 160, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 160, 'kernel_size': [2, 2], 'strides': (1, 1)}]]

    # Max Pool inputs
    #     pool_size : An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window
    #                 Can be a single integer to specify the same value for all spatial dimensions
    #     strides : n integer or tuple/list of 2 integers, specifying the strides of the pooling operation
    #               Can be a single integer to specify the same value for all spatial dimensions
    max_pool_specifications = [[{'use': False, 'pool_size': [1, 2], 'strides': [1, 1]},
                                {'use': True, 'pool_size': [1, 4], 'strides': [1, 1]}],
                               [{'use': False, 'pool_size': [1, 2], 'strides': [1, 1]},
                                {'use': False, 'pool_size': [1, 2], 'strides': [1, 1]},
                                {'use': True, 'pool_size': [2, 4], 'strides': [1, 1]}],
                               [{'use': False, 'pool_size': [1, 2], 'strides': [1, 1]},
                                {'use': False, 'pool_size': [1, 2], 'strides': [1, 1]},
                                {'use': True, 'pool_size': [2, 4], 'strides': [1, 1]}]]

    # Dropout inputs
    #     use : to use dropout in this layer
    #     rate : dropout rate
    dropout_parameters = [[{'use': True, 'rate': 0.5},
                           {'use': True, 'rate': 0.5}]]

    model = DeepCas(sess=session,
                    data_shape=data_shape,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    conv_parameters=conv2d_specifications,
                    max_pool_parameters=max_pool_specifications,
                    dropout_parameters=dropout_parameters,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                    tensorboard_directory=tensorboard_directory)

    # Data Manip
    #-------------------------------------------------------------------------
    path = ['data\Haeussler.tsv']
    training_stacked, training_score, test_stacked, test_score = getData(path=path,
                                                                         sub_sample=sub_sample,
                                                                         val_size=val_size)

    #-------------------------------------------------------------------------

    model.input_data(training_stacked, training_score, test_stacked, test_score)

    print('> directory: {0}'.format(tensorboard_directory))
    print('> Begin Training!')
    print('--------------------------------------------------------')
    model.train(isRestore=restore)


def processData(path):
    '''
    Data Importing:
    ---------------
    Takes input data from LAI & YU2 guideRNA to targetDNA binding scores from
    the MITEstimator and splits it into three lists guide, target, and score

    guide : list of lists
        onehot encoded gRNA sequences

    target : list of lists
        one hot encoded targetDNA sequences

    scores : list of floats
        numeric scores from the MITEstimator
    '''

    p = Processing()
    data = p.getData(path)
    guide, target, score = p.splitData(data)

    i = 1
    # print('Guide: {0}\tTarget: {1}\tScore: {2}'.format(guide[i], target[i], scores[i]))

    target = [target[i][30:53] for i in range(len(target))]

    print('> Sample Row:\n\tGuide:  {0}\n\tTarget: {1}\n\tScore:  {2}'.format(guide[i], target[i], score[i]))

    print('> number of sample: {0}'.format(len(guide)))
    print('> filtering out pam')
    guideHasPam = len(guide) == sum([guide[i][-2:] in 'GG' for i in range(len(guide))])
    print('\t> all guides have pam: {0}'.format(guideHasPam))
    if not guideHasPam:
        l = []
        for i in range(len(guide)):
            if guide[i][-2:] not in 'GG':
                l.append(guide[i] + target[i][-3:])
            else:
                l.append(guide[i])
        guide = l

        guide_wrong_len = []
        for i in range(len(guide)):
            if len(guide[i]) is not 23:
                guide_wrong_len.append(i)
        print('\t> removing seqs: {0}'.format(len(guide_wrong_len)))
        for i in reversed(guide_wrong_len):
            del guide[i]
            del target[i]
            del score[i]

    guideHasPam = len(guide) == sum([guide[i][-2:] in 'GG' for i in range(len(guide))])
    targetHasPam = len(target) == sum([target[i][-2:] in 'GG' for i in range(len(target))])

    print('\t> all guides have pam: {0}'.format(guideHasPam))
    print('\t> all targets have pam: {0}'.format(targetHasPam))

    print('> number of sample: {0}'.format(len(guide)))
    # guide, target, score = doNothing(guide, target, score)

    print('> percent same in guide & target: {0}'.format(numSame(guide, target)/len(guide) * 100))
    guide = p.oneHotEncoderList(guide)
    target = p.oneHotEncoderList(target)

    guide = np.array(guide)
    target = np.array(target)
    score = np.array(score)

    return guide, target, score


def splitData(val_size, guide, target, score):
    assert val_size < len(score)

    training_idx = np.random.randint(len(score), size=len(score)-val_size)
    test_idx = np.random.randint(len(score), size=val_size)

    training_guide, test_guide = np.array(guide[training_idx, :]),  np.array(guide[test_idx, :])
    training_target, test_target = np.array(target[training_idx, :]),  np.array(target[test_idx, :])

    training_score, test_score = np.array([score[i] for i in training_idx]),  np.array([score[i] for i in test_idx])

    return training_guide, training_target, training_score, test_guide, test_target, test_score


def getData(path, sub_sample, val_size):

    path = ['data\Haeussler.tsv']

    guide, target, score = processData(path)

    if sub_sample:
        guide, target, score = randomSample(sub_sample, guide, target, score)

    '''
    Shapes:
        stacked = [Size, 4, 23, 2]
        score = [Size]
    '''

    training_guide, training_target, training_score, test_guide, test_target, test_score = splitData(val_size, guide, target, score)

    training_stacked = np.stack((training_guide, training_target), axis=-1)
    test_stacked = np.stack((test_guide, test_target), axis=-1)

    print('> total data:{}'.format(str(len(score)).rjust(10, ' ')))
    print('> training data:{}'.format(str(len(training_score)).rjust(10, ' ')))
    print('> test data:{}'.format(str(len(test_score)).rjust(10, ' ')))

    print('> stacked: {}, {}'.format(str(list(training_stacked.shape)).rjust(10, ' '), str(list(test_stacked.shape)).rjust(10, ' ')))
    print('> score: {}, {}'.format(str(list(training_score.shape)).rjust(9, ' '), str(list(test_score.shape)).rjust(9, ' ')))

    return training_stacked, training_score, test_stacked, test_score


def numSame(list1, list2):
    assert len(list1) == len(list2)
    nSame = 0
    for i in range(len(list1)):
        if str(list1[i]) in str(list2[i]):
            nSame += 1
    return nSame


def randomSample(size, guide, target, score):

    idx = np.arange(0, len(score))
    np.random.shuffle(idx)

    idx = idx[:size]
    guide = np.array([guide[i] for i in idx])
    target = np.array([target[i] for i in idx])
    score = np.array([score[i] for i in idx])

    return guide, target, score


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    print('-------------------------------------------------------------------')
    main()
