import torch
import pandas as pd
import os
import os.path as osp
import json

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)


def load_file(filename):
    data = None
    assert osp.exists(filename)
    file_type = osp.splitext(filename)[1]
    if file_type == '.csv':
        data = pd.read_csv(filename)
    elif file_type == '.json' or file_type == '.txt':
        with open(filename, 'r') as fp:
            if file_type == '.json':
                data = json.load(fp)
            if file_type == '.txt':
                data = fp.readlines()
                data = [d.rstrip('\n') for d in data]
    return data

def save_file(filename, data):

    print('save to {}'.format(filename))
    dirname = osp.dirname(filename)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'w') as fp:
        json.dump(data, fp)
