import numpy as np
import csv

import tensorflow as tf

from processing import Processing
from DeepCas import DeepCas


def main():

    # Model Paramaters
    #-------------------------------------------------------------------------

    session = tf.Session()
    data_shape = [4, 23, 2]
    batch_size = 1000
    epochs = 100000
    learning_rate = 1e-4

    sub_sample = 35000

    # Conv2d inputs
    #     filters : Integer, dimensionality of the output space (ie. the number of filters in the convolution)
    #     kernel_size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    #                   Can be a single integer to specify the same value for all spatial dimensions
    #     strides : An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    #               Can be a single integer to specify the same value for all spatial dimensions
    conv2d_specifications = [[{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}],
                             [{'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)},
                              {'filters': 80, 'kernel_size': [2, 2], 'strides': (1, 1)}]]

    # Max Pool inputs
    #     pool_size : An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window
    #                 Can be a single integer to specify the same value for all spatial dimensions
    #     strides : n integer or tuple/list of 2 integers, specifying the strides of the pooling operation
    #               Can be a single integer to specify the same value for all spatial dimensions
    max_pool_specifications = [[{'pool_size': [1, 1], 'strides': [1, 1]},
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

    paths = ['data\LAI_numeric.csv', 'data\YU2_numeric.csv']
    pam_path = 'data/pam.csv'
    proto_path = 'data/protospacer.csv'

    guide, target, score = processData(paths, pam_path, proto_path)

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
    print('Stacked: {}'.format(str(list(stacked.shape)).rjust(10, ' ')))
    print('Score: {}'.format(str(list(score.shape)).rjust(10, ' ')))

    #-------------------------------------------------------------------------

    model.input_data(stacked, score)

    model.train(tb_directory)


def processData(data_paths, pam_path, proto_path):
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
    # get list of raw data
    data = p.getData(data_paths)

    # split the data into three lists, gRNA, target seq, and their corresponding MITScore
    guide, target, score = p.splitData(data)

    # get a list of pam seqs
    pam = []
    with open(pam_path, 'r') as f:
        data_itter = csv.reader(f)
        pam = [data for data in data_itter]

    # get a list of protospacer sequences
    proto = []
    with open(proto_path, 'r') as f:
        data_itter = csv.reader(f)
        proto = [data for data in data_itter]

    # combine the protospacer & pam into a dictionary
    proto_pam_combo = {}  # {{proto[i][1]: pam[i][1]} for i in range(len(pam))}
    for i in range(len(pam)):
        proto_pam_combo.update({proto[i][1]: pam[i][1]})

    # append the pam sequence to it's corresponding gRNA sequence
    guide_full = []
    for g in guide:
        try:
            guide_full.append(str(g) + str(proto_pam_combo[g]))
        except:
            continue

    # onehot encode the sequences
    guide = p.oneHotEncoderList(guide_full)
    target = p.oneHotEncoderList(target)

    guide = np.array(guide)
    target = np.array(target)
    score = np.array(score)  # Shape: [Size]

    return guide, target, score


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
