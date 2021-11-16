# ====================================================
# @Time    : 9/17/20 8:43 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : split_data.py
# ====================================================
from datautils import utils
import os.path as osp
import pandas as pd
import random as rd


def split_samples(samples, train_videos, test_videos, val_videos):
    num = len(samples)
    train_list = []
    val_list = []
    test_list =  []
    print(num)
    # print(type(samples))
    for i in range(num):
        sample = samples.loc[i].astype('string')
        vname = sample['video']
        # print(vname)
        if vname in train_videos:
            train_list.append(sample)
        elif vname in val_videos:
            val_list.append(sample)
        else:
            test_list.append(sample)
        if i % 1000 == 0:
            print(i, num)

    print('Train: {}, Val: {}, Test: {}'.format(len(train_list), len(val_list), len(test_list)))
    rd.shuffle(train_list)
    rd.shuffle(val_list)
    rd.shuffle(test_list)

    df_test = pd.DataFrame(test_list)
    df_test.to_csv('dataset/VidQA/test.csv', sep=',')
    df_train = pd.DataFrame(train_list)
    df_train.to_csv('dataset/VidQA/train.csv', sep=',')
    df_val = pd.DataFrame(val_list)
    df_val.to_csv('dataset/VidQA/val.csv', sep=',')



def main():

    split_dir = '../annos/{}.json'
    train_videos = utils.load_file(split_dir.format('train'))
    test_videos = utils.load_file(split_dir.format('test'))
    val_videos = utils.load_file(split_dir.format('val'))
    name = 'dataset/VidQA/all.csv'
    samples = pd.read_csv(name, index_col=0, delimiter=' ')
    split_samples(samples, train_videos, test_videos, val_videos)

if __name__== "__main__":
    main()