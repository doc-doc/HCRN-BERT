# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
import os.path as osp

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class VideoQADataset(Dataset):

    def __init__(self, answers, video_ids, q_ids,
                 app_feat_dir, mot_feat_dir, mode):

        self.all_answers = answers

        self.all_video_ids = video_ids
        self.all_q_ids = q_ids
        self.app_feat_dir = app_feat_dir
        self.mot_feat_dir = mot_feat_dir

        self.question_type = 'mulchoices'

        bert_path = '../data/feats/qas_bert/'
        self.bert_file = osp.join(bert_path, 'bert_ft_{}.h5'.format(mode))


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None

        with h5py.File(self.bert_file, 'r') as fp:
            question = fp['feat'][index]


        question_len = []
        for r in range(question.shape[0]):
            question_len.append(nonzero_row(question[r]))

        question = torch.from_numpy(question).type(torch.float32)
        question_len = torch.LongTensor(question_len)

        video_idx = self.all_video_ids[index]
        question_idx = self.all_q_ids[index]

        app_feat_file = osp.join(self.app_feat_dir, str(video_idx)+'.npy')
        mot_feat_file = osp.join(self.mot_feat_dir, str(video_idx)+'.npy')

        appearance_feat = np.load(app_feat_file)
        motion_feat = np.load(mot_feat_file)
        


        return (video_idx, question_idx, answer,
                appearance_feat, motion_feat, question, question_len)

    def __len__(self):
        return len(self.all_q_ids)



def nonzero_row(A):
    i = 0
    for row in A:
        if row.sum() == 0:
            break
        i += 1

    return i


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            # questions = obj['questions']
            # questions_len = obj['questions_len']
            # print('haha', questions_len.shape)
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            # glove_matrix = obj['glove']
            # ans_candidates = np.zeros(5)
            # ans_candidates_len = np.zeros(5)
            # if question_type in ['action', 'transition']:
            #     ans_candidates = obj['ans_candidates']
            #     ans_candidates_len = obj['ans_candidates_len']

        mode = 'train'
        if 'train_num' in kwargs:
            mode = 'train'
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                # questions = questions[:trained_num]
                # questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                # if question_type in ['action', 'transition']:
                #     ans_candidates = ans_candidates[:trained_num]
                #     ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            mode = 'val'
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                # questions = questions[:val_num]
                # questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                # if question_type in ['action', 'transition']:
                #     ans_candidates = ans_candidates[:val_num]
                #     ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            mode = 'test'
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                # questions = questions[:test_num]
                # questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                # if question_type in ['action', 'transition']:
                #     ans_candidates = ans_candidates[:test_num]
                #     ans_candidates_len = ans_candidates_len[:test_num]

        # print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        # with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
        #     app_video_ids = app_features_file['ids'][()]
        # app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        #
        # print('loading motion feature from %s' % (kwargs['motion_feat']))
        # with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
        #     motion_video_ids = motion_features_file['ids'][()]
        # motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feat_dir = kwargs.pop('app_feat_dir')
        self.mot_feat_dir = kwargs.pop('mot_feat_dir')

        self.dataset = VideoQADataset(answers, video_ids, q_ids, self.app_feat_dir, self.mot_feat_dir, mode)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        # self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
