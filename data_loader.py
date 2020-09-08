import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from configuration import config


class ImageFolder(data.Dataset):
    def __init__(self, image_path, GT_path, image_size, mode='train'):
        """Initializes image paths, including each image path"""
        self.image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
        self.GT_paths = list(map(lambda x: os.path.join(GT_path, x), os.listdir(GT_path)))
        self.image_size = image_size
        self.mode = mode
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        if self.mode == 'test':
            fid = open('datasets' + '/test_patch_read_order.txt', 'a')
            fid.write(self.image_paths[index] + '\n')
            fid.close()
        single_image_path = self.image_paths[index]  # single image
        single_GT_path = self.GT_paths[index]



        image = Image.open(single_image_path).convert('L')
        GT = Image.open(single_GT_path).convert('1')

        Transform = []
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5,), (0.5,))
        image = Norm_(image)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, GT_path, image_size, batch_size, num_workers=8, mode='train'):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(image_path=image_path, GT_path=GT_path, image_size=image_size, mode=mode)
    shuffle = True
    if mode == 'test':
        shuffle = False
        num_workers = 1
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return data_loader
