import argparse
import numpy as np
import os

from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from datautils import nextqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nextqa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'nextqa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt', default = 'dataset/glove/glove.840.300d.pkl',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='dataset/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='dataset/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='transition')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'tgif-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/tgif-qa/csv/{}_{}_question.csv'
        args.output_pt = 'dataset/tgif-qa/{}/tgif-qa_{}_{}_questions.pt'
        args.vocab_json = 'dataset/tgif-qa/{}/tgif-qa_{}_vocab.json'
        # check if dataset folder exists
        if not os.path.exists('dataset/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('dataset/tgif-qa/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended(args)
        else:
            tgif_qa.process_questions_mulchoices(args)
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msrvtt/annotations/{}_qa.json'.format(args.mode)
        # check if dataset folder exists
        if not os.path.exists('dataset/{}'.format(args.dataset)):
            os.makedirs('dataset/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if dataset folder exists
        if not os.path.exists('dataset/{}'.format(args.dataset)):
            os.makedirs('dataset/{}'.format(args.dataset))
        msvd_qa.process_questions(args)
    if args.dataset == 'nextqa':
        args.annotation_file = 'dataset/nextqa/{}.csv'
        args.output_pt = 'dataset/nextqa/{}.pt'
        args.vocab_json = 'dataset/nextqa/{}_vocab.json'
        # check if dataset folder exists
        # if not os.path.exists('dataset/VidQA/{}'.format(args.question_type)):
        #     os.makedirs('dataset/VidQA/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            nextqa.process_questions_openended(args)
        else:
            nextqa.process_questions_mulchoices(args)
