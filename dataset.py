import os
import random
import shutil
from shutil import copyfile
from configuration import config
import numpy as np
import preprocessing
from preprocessing import patch_dataset


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)

def data_sythesis(config, mkdir=False, patch_method=True):
    if mkdir:
        rm_mkdir(config['original_data_path'])
        rm_mkdir(config['original_GT_path'])
        rm_mkdir(config['train_path'])
        rm_mkdir(config['train_GT_path'])
        rm_mkdir(config['valid_path'])
        rm_mkdir(config['valid_GT_path'])
        rm_mkdir(config['test_path'])
        rm_mkdir(config['test_GT_path'])

    num_total, data_list, GT_list = preprocessing.data_preprocess(config)

    # split datasets/groundtruth into training/validation/testing set
    num_train = int(config['train_ratio'] * num_total)
    num_valid = int(config['valid_ratio'] * num_total)
    num_test = num_total - num_train - num_valid

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()

        src = os.path.join(config['original_data_path'], data_list[idx])
        dst = os.path.join(config['train_path'], data_list[idx])
        copyfile(src, dst)
        src = os.path.join(config['original_GT_path'], GT_list[idx])
        dst = os.path.join(config['train_GT_path'], GT_list[idx])
        copyfile(src, dst)

    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config['original_data_path'], data_list[idx])
        dst = os.path.join(config['valid_path'], data_list[idx])
        copyfile(src, dst)
        src = os.path.join(config['original_GT_path'], GT_list[idx])
        dst = os.path.join(config['valid_GT_path'], GT_list[idx])
        copyfile(src, dst)

    for i in range(num_test):
        idx = Arange.pop()

        src = os.path.join(config['original_data_path'], data_list[idx])
        dst = os.path.join(config['test_path'], data_list[idx])
        copyfile(src, dst)
        src = os.path.join(config['original_GT_path'], GT_list[idx])
        dst = os.path.join(config['test_GT_path'], GT_list[idx])
        copyfile(src, dst)

    if patch_method:
        rm_mkdir(config['train_patch_path'])
        rm_mkdir(config['train_GT_patch_path'])
        rm_mkdir(config['valid_patch_path'])
        rm_mkdir(config['valid_GT_patch_path'])
        rm_mkdir(config['test_patch_path'])
        rm_mkdir(config['test_GT_patch_path'])

        patch_size = config['patch_size']
        img_size = config['resize']

        # sample patches from training set
        config['mode'] = 'train'
        num_train_patches = int(config['num_patches']*2/3)
        patch_dataset(config['train_path'], config['train_GT_path'], img_size,
                          patch_size, config['mode'], num_patches=num_train_patches)

        # sample patches from validation set
        config['mode'] = 'valid'
        num_valid_patches = config['num_patches'] - num_train_patches
        patch_dataset(config['valid_path'], config['valid_GT_path'], img_size,
                      patch_size, config['mode'], num_patches=num_valid_patches)

        # sample patches from testing set
        config['mode'] = 'test'
        patch_dataset(config['test_path'], config['test_GT_path'], img_size,
                      patch_size, config['mode'], num_patches=None)
