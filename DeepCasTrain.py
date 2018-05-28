import numpy as np
import csv

import tensorflow as tf

from processing_haeussler import Processing
from DeepCas import DeepCas


def main():

    # Model Paramaters
    #-------------------------------------------------------------------------

    session = tf.Session()
    data_shape = [4, 23, 2]
    batch_size = 1500
    epochs = 100000
    learning_rate = 1e-4

    sub_sample = None

    # Conv2d inputs
    #     filters : Integer, dimensionality of the output space (ie. the number of filters in the convolution)
    #     kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    #                   Can be a single integer to specify the same value for all spatial dimensions
    #     strides : An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    #               Can be a single integer to specify the same value for all spatial dimensions
    conv2d_specifications = [[{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 40, 'kernel_size': [2, 2], 'strides': (1, 1)}]]

    # Max Pool inputs
    #     pool_size : An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window
    #                 Can be a single integer to specify the same value for all spatial dimensions
    #     strides : n integer or tuple/list of 2 integers, specifying the strides of the pooling operation
    #               Can be a single integer to specify the same value for all spatial dimensions
    max_pool_specifications = [[{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}],
                               [{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}],
                               [{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}],
                               [{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}],
                               [{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}],
                               [{'pool_size': [1, 1], 'strides': [1, 1]},
                                {'pool_size': [1, 1], 'strides': [1, 1]}]]

    save_directory = './testing/saved_model/'

    tb_directory = './testing/tensorboard/'

    model = DeepCas(sess=session,
                    data_shape=data_shape,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    conv_parameters=conv2d_specifications,
                    max_pool_parameters=max_pool_specifications,
                    save_directory=save_directory)

# Data Manip
#-------------------------------------------------------------------------

    path = ['data\Haeussler.tsv']

    guide, target, score = processData(path)

    if sub_sample:
        guide, target, score = randomSample(sub_sample, guide, target, score)

    '''
    Shapes:
        stacked = [Size, 4, 23, 2]
        score = [Size]
    '''

    stacked = np.stack((guide, target), axis=-1)

    print('> input data sizes:\n\tguides: {}\n\ttargets: {}\n\tscores: {}'.format(
        str(len(guide)).rjust(8, ' '),
        str(len(target)).rjust(7, ' '),
        str(len(score)).rjust(8, ' ')))
    print('> Stacked: {}'.format(str(list(stacked.shape)).rjust(10, ' ')))
    print('> Score: {}'.format(str(list(score.shape)).rjust(9, ' ')))

    #-------------------------------------------------------------------------

    model.input_data(stacked, score)

    print('> Begin Training!')
    print('--------------------------------------------------------')
    model.train(tb_directory)


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
    #guide, target, score = doNothing(guide, target, score)

    print('> percent same in guide & target: {0}'.format(numSame(guide, target)/len(guide) * 100))
    guide = p.oneHotEncoderList(guide)
    target = p.oneHotEncoderList(target)

    guide = np.array(guide)
    target = np.array(target)
    score = np.array(score)

    return guide, target, score


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
    print('---------------------------------------------------------------------')
    main()
