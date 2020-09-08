import os
from solver import Solver, train_accuracy, validation_accuracy
from data_loader import get_loader
from torch.backends import cudnn
import random
from configuration import config
import dataset
import matplotlib.pyplot as plt
import numpy as np

def main(config):
    cudnn.benchmark = True

    # Create model and result directories if not exist
    #dataset.data_sythesis(config, mkdir=True)
    if not os.path.exists(config['model_path']):
        dataset.rm_mkdir(config['model_path'])
    dataset.rm_mkdir(config['result_path'])
    config['result_path'] =  os.path.join(config['result_path'], config['model_type'])
    dataset.rm_mkdir(config['result_path'])

    lr = 0.005
    epoch = 10 #random.choice([1, 2])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config['num_epochs'] = epoch
    config['lr'] = lr
    config['num_epochs_decay'] = decay_epoch

    print(config)

    # patch-based method
    train_loader = get_loader(image_path=config['train_patch_path'], GT_path=config['train_GT_patch_path'],
                              image_size=config['patch_size'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='train')
    valid_loader = get_loader(image_path=config['valid_patch_path'], GT_path=config['valid_GT_patch_path'],
                              image_size=config['patch_size'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='valid')
    test_loader = get_loader(image_path=config['test_patch_path'], GT_path=config['test_GT_patch_path'],
                              image_size=config['patch_size'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='test')

    """
    train_loader = get_loader(image_path=config['train_path'], GT_path=config['train_GT_path'],
                              image_size=config['resize'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='train')
    valid_loader = get_loader(image_path=config['valid_path'], GT_path=config['valid_GT_path'],
                              image_size=config['resize'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='valid')
    test_loader = get_loader(image_path=config['test_path'], GT_path=config['test_GT_path'],
                              image_size=config['resize'], batch_size=config['batch_size'],
                              num_workers=config['num_workers'], mode='test')
    """
    solver = Solver(config, train_loader, valid_loader, test_loader)
    solver.train()
    plt.plot(np.arange(1, epoch + 1), train_accuracy, label='train accuracy')
    plt.plot(np.arange(1, epoch + 1), validation_accuracy, label='train accuracy')
    plt.savefig('plot.png', dpi=300)
    solver.test()


if __name__ == '__main__':
    main(config)