# /processing.py

import os
import csv
import numpy as np
from pathlib import Path


class Processing():
    def __init__(self, log=True):
        '''
        log : boolean
            log what is going on to the console
        '''
        self._DNA = ['A', 'T', 'C', 'G']
        self.log = log

    def _isPathExists(self, path):
        '''
        Inputs:
        -------
        path : string
            path to a file

        Returns:
        --------
        boolean
            if path exists True
            else False
        '''

        return Path(path).exists()

    def _isDNA(self, seq):
        '''
        Inputs:
        -------
        seq : string
            DNA sequence

        Returns:
        --------
        boolean
            if is DNA True
            else False
        '''
        return all([nucleotide in self._DNA for nucleotide in seq])

    def _getFileName(self, path):
        '''
        Inputs:
        -------
        path : string
            path to a file

        Returns:
        --------
        string
            filename from a path
            ie. example.csv
        '''

        return os.path.basename(path)

    def getData(self, paths):
        '''
        Inputs:
        -------
        paths : list of strings
            list of paths to data file locations

        Returns:
        --------
        data_all: list of lists of strings & integers
            outputs the data from all the files in one large consecutive list
        '''

        data_all = []
        for path in paths:
            if not self._isPathExists(path):
                raise PathError('Path does not exist: {}'.format(path))

            else:
                if self.log:
                    print('> importing data from: {}'.format(self._getFileName(path)))

                # open the file and get all the data
                with open(path, 'r') as file:
                    data_itter = csv.reader(file, delimiter='\t')
                    # ignore data that is not dna | ie. header if self._isDNA(data[2]) and self._isDNA(data[6])
                    data = [data for data in data_itter if self._isDNA(data[2]) and self._isDNA(data[6])]

                # append the data to a list
                data_all.append(data)

        # Concatinate the data from different files
        data_all = sum(data_all, [])

        return data_all

    def splitData(self, data, log=False):
        '''
        Inputs:
        -------
        data : list of lists of strings & integers
            list of data [index, index, gRNA, target, score]

        Returns:
        --------
        guide : list of strings
            guideRNAs

        target : list of strings
            target sequences

        score : list of integers
            MITEstimator binding scores
        '''

        if self.log:
            print('> splitting dataset by columns')

        guide, target, score = [], [], []
        for row in data:
            guide.append(row[2])
            target.append(row[6])
            score.append(row[3])

        return guide, target, score

    def oneHotEncoder(self, seq):
        '''
        Inputs:
        -------
        seq : string
            DNA sequence

        Returns:
        --------
        oh : list of list of integers
            one hot encoded DNA sequence

        Example:
        --------
        Input :
            ATATCG
        Output :
            A:    [[1, 0, 1, 0],
            T:     [0, 1, 0, 1],
            C:     [0, 0, 0, 0, 1,0]
            G:     [0, 0, 0, 0,0,1]]
        '''

        # Ensure the sequence is uppercase
        seq = seq.upper()

        # Init the onehot to 0's
        oh = [[0 for _ in range(len(seq))] for _ in range(len(self._DNA))]

        # Find positions in the sequence that have A,T,C,G and output replace with 1
        for i, base in enumerate(self._DNA):
            positions = self._find(base, seq)
            for p in positions:
                oh[i][p] = 1

        return oh

    def oneHotEncoderList(self, seqs):
        '''
        Inputs:
        -------
        seqs : list of lists of strings
            sequences to encode

        Returns:
        --------
        onehot_encoded : list of lists of lists of integers
            one hot encoded sequences
            ie.
                [[[0/1][0/1][0/1][0/1]]
                    ...
                 [[0/1][0/1][0/1][0/1]]]
        '''

        if self.log:
            print('> onehot encoding seqs of length: {}'.format(len(seqs[0])))

        onehot_encoded = []
        for seq in seqs:
            onehot_encoded.append(self.oneHotEncoder(seq))

        return onehot_encoded

    def oneHotDecoder(self, oh):
        '''
        Inputs:
        -------
        oh : list of lists of integers
            onehot encoded sequence

        Returns:
        --------
        seq : string
            sequence decoded from the onehot input

        Example:
        --------
        Input:
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        Output:
            ATCG
        '''

        seq = ''
        for i in range(len(oh[0])):
            if oh[0][i] is 1:
                seq += 'A'
            elif oh[1][i] is 1:
                seq += 'T'
            elif oh[2][i] is 1:
                seq += 'C'
            elif oh[3][i] is 1:
                seq += 'G'

        return seq

    def oneHotDecoderList(self, ohs):
        '''
        Inputs:
        -------
        ohs : list of lists of lists of integers
            list of onehot encoded sequence

        Results:
        --------
        seqs : list of strings
            decoded sequences
        '''

        if self.log:
            print('> onehot decoding seqs of length: {}'.format(len(seqs[0])))

        seqs = []
        for oh in ohs:
            seqs.append(self.oneHotDecoder(oh))

        return seqs

    def _find(self, base, seq):
        '''
        Inputs:
        -------
        base : character
            nucleotide to search for

        seq : string
            sequence to search in

        Results:
        --------
        positions : list of integers
            positions where the nucleotide is in the sequence
        '''

        positions = []
        for i, nucleotide in enumerate(seq):
            if base in nucleotide:
                positions.append(i)

        return positions


class PathError(ValueError):
    '''
    Error to throw when path does not exist
    '''

    def __init__(self, *args, **kwargs):
        ValueError.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    path = ['data\Haeussler.tsv']
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
                l.append(guide[i] + target[i][-2:])
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

    def numSame(list1, list2):
        assert len(list1) == len(list2)
        nSame = 0
        for i in range(len(list1)):
            if str(list1[i]) in str(list2[i]):
                nSame += 1
        return nSame

    print('> percent same in guide & target: {0}'.format(numSame(guide, target)/len(guide) * 100))

    guide = p.oneHotEncoderList(guide)
    target = p.oneHotEncoderList(target)

    guide = np.array(guide)
    target = np.array(target)
    score = np.array(score)

    # for i in range(10):
    #    print('Guide: {0}\tTarget: {1}\tScore: {2}'.format(guide[i], target[i], scores[i]))
