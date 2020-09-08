from PIL import Image
import numpy as np
import os
from configuration import config

def merge_image(bottom_img, top_img):
    """
    :param bottom_img: image at bottom
    :param top_img: image lies on the bottom image
    :param save_img: result image path to save image
    :return:
    """
    img1 = np.array(bottom_img)  # Take two images and convert to numpy array
    img2 = np.array(top_img)
    merge_img = img1 + img2
    PIL_image = Image.fromarray(merge_img)
    return PIL_image


def reshape_image(image, img_size):  # resize the image
    return image.resize((img_size, img_size), Image.ANTIALIAS)


def sample_image(img, patch_size, pos_row, pos_col):
    """
    sample a patch image with size 48 x 48
    :param img: original image
    :param patch_size:
    :param pos_row: top_left position row index of patch image ranging from [0, 815] (863 - 48)
    :param pos_col: top_left position col index of patch image ranging from [0, 815] (863 - 48)
    :return: patch image
    """
    img = np.array(img)
    patch_img = img[pos_row: pos_row + patch_size, pos_col: pos_col + patch_size]
    return Image.fromarray(patch_img)


def patch_dataset(datapath, label_path, img_size, patch_size, mode, num_patches=None):
    """
    sample patches of images used for training / testing
    :param datapath: (string) contains image sets to be patched
    :param num_patches: number of patches
    :param testset: if true, patch testsset not one by one, not randomly
    :return:
    """
    filename = os.listdir(datapath)
    num_file = len(filename)

    if mode == 'train':
        while num_patches >= 0:
            img_idx = int(np.random.randint(0, num_file, 1))  # choose an image randomly to sample patches
            img = Image.open(datapath + filename[img_idx])
            img_label = Image.open(label_path + filename[img_idx])  # groundtruth image
            row, col = np.random.randint(0, img_size - patch_size + 1, size=2)

            patch_img = sample_image(img, patch_size, row, col)
            patch_label = sample_image(img_label, patch_size, row, col)
            save_path_patch = os.path.join(config['train_patch_path'], '{}.tif'.format(num_patches))
            save_path_patch_gt = os.path.join(config['train_GT_patch_path'], '{}.tif'.format(num_patches))
            patch_img.save(save_path_patch)
            patch_label.save(save_path_patch_gt)
            num_patches = num_patches - 1

    elif mode == 'valid':
        while num_patches >= 0:
            img_idx = int(np.random.randint(0, num_file, 1))  # choose an image randomly to sample patches
            img = Image.open(datapath + filename[img_idx])
            img_label = Image.open(label_path + filename[img_idx])  # groundtruth image
            row, col = np.random.randint(0, img_size - patch_size + 1, size=2)

            patch_img = sample_image(img, patch_size, row, col)
            patch_label = sample_image(img_label, patch_size, row, col)
            save_path_patch = os.path.join(config['valid_patch_path'], '{}.tif'.format(num_patches))
            save_path_patch_gt = os.path.join(config['valid_GT_patch_path'], '{}.tif'.format(num_patches))
            patch_img.save(save_path_patch)
            patch_label.save(save_path_patch_gt)
            num_patches = num_patches - 1

    else:  # test
        num = 0
        for idx in range(len(filename)):
            img = Image.open(datapath + filename[idx])
            img_label = Image.open(label_path + filename[idx])
            for row in np.arange(0, img_size - patch_size + 1, patch_size):
                for col in np.arange(0, img_size - patch_size + 1, patch_size):
                    patch_img = sample_image(img, patch_size, row, col)
                    patch_label = sample_image(img_label, patch_size, row, col)
                    save_path_patch = os.path.join(config['test_patch_path'], '{}.tif'.format(num))
                    save_path_patch_gt = os.path.join(config['test_GT_patch_path'], '{}.tif'.format(num))
                    patch_img.save(save_path_patch)
                    patch_label.save(save_path_patch_gt)
                    num = num + 1



def data_preprocess(config):
    img_size = config['resize']
    dir_data_path = 'D:/Heidelberg/Semester2/project_bioCV/datasets/rawdata/'
    dir_gt_path = 'D:/Heidelberg/Semester2/project_bioCV/datasets/groundtruth/'
    data_list = []
    GT_list = []
    filenames = os.listdir(dir_gt_path)
    filenames_data = os.listdir(dir_data_path)

    # preprocess groundtruth
    for i in range(len(filenames)):
        newfile_gt = filenames[i].split()[-1][8:]
        if len(newfile_gt) == 12:
            newfile_gt = newfile_gt[::-1] + '0'
            newfile_gt = newfile_gt[::-1]
        if not os.path.exists(dir_gt_path + filenames[i]):
            os.rename(dir_gt_path + filenames[i], dir_gt_path + newfile_gt)
    filenames = sorted(os.listdir(dir_gt_path))

    # preprocess rawdata
    for idx in range(len(filenames_data)):
        image = Image.open(os.path.join(dir_data_path, filenames_data[idx])).convert('L')
        image = reshape_image(image, img_size)
        save_path = config['original_data_path'] + '{}.tif'.format(idx)
        data_list.append('{}.tif'.format(idx))
        image.save(save_path)

    # merge and resize images
    subfile1 = [filenames[i] for i in range(len(filenames)) if i % 2 == 0]
    subfile2 = [filenames[i] for i in range(len(filenames)) if i % 2 != 0]
    for idx, (path1, path2) in enumerate(zip(subfile1, subfile2)):
        top_img = Image.open(dir_gt_path + path1)
        bottom_img = Image.open(dir_gt_path + path2)
        merged_img = merge_image(bottom_img, top_img)
        image_reshaped = reshape_image(merged_img, img_size)
        save_path = config['original_GT_path'] + '{}.tif'.format(idx)
        GT_list.append('{}.tif'.format(idx))
        image_reshaped.save(save_path)
    return len(filenames_data), data_list, GT_list




