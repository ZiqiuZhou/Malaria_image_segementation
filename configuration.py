config = {
    # data pathes
    'original_data_path': 'datasets/rawdata/',
    'original_GT_path': 'datasets/gt/',
    'train_path': 'datasets/train/',
    'train_GT_path': 'datasets/train_GT/',
    'valid_path': 'datasets/valid/',
    'valid_GT_path': 'datasets/valid_GT/',
    'test_path': 'datasets/test/',
    'test_GT_path': 'datasets/test_GT/',
    'train_patch_path': 'datasets/train_patch/',
    'train_GT_patch_path': 'datasets/train_GT_patch/',
    'valid_patch_path': 'datasets/valid_patch/',
    'valid_GT_patch_path': 'datasets/valid_GT_patch/',
    'test_patch_path': 'datasets/test_patch/',
    'test_GT_patch_path': 'datasets/test_GT_patch/',
    'model_path': 'models/',
    'result_path': 'results/',

    # model parameters
    'num_patches': 20000,
    'patch_size': 96,
    'resize': 864,
    'model_type': 'R2U_Net',
    't': 2,  # time steps for RNN layer

    # training parameters
    'train_ratio': 0.6,
    'valid_ratio': 0.2,
    'test_ratio': 0.2,
    'batch_size': 32,
    'img_ch': 1,
    'output_ch': 1,
    'num_epochs': 10,
    'num_epoches_decay': 70,
    'num_workers': 8,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'log_step': 2,
    'val_step': 2,

    # other
    'mode': 'train'
}
