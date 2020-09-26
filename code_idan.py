import torch
import torch.utils.data
import torchvision
import skimage.io
import skimage.transform
import os
import re
import sys
import math
import random
import torchvision.transforms


def generate_training_set(dataset_root_dir, train_ratio=1, batch_size=4):
    ellipse_dataset = EllipseDataSet(dataset_root_dir=dataset_root_dir)
    #print("len(ellipse_dataset is:" + str(len(ellipse_dataset)))
    #print("indexes range is: " + str(range(len(ellipse_dataset))))

    data_indexes = list(range(len(ellipse_dataset)))
    #print(data_indexes)

    #random.shuffle(data_indexes)


    # training_indexes = data_indexes[:math.floor(len(data_indexes)*train_ratio)]
    #training_indexes = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350] #
    training_indexes = [0, 1, 2, 3]
    print("train indexes are:" + str(training_indexes))
    # test_indexes = data_indexes[math.floor(len(data_indexes)*train_ratio):]
    test_indexes = []
    print("test indexes are:" + str(test_indexes))

    training_sampler = torch.utils.data.sampler.SubsetRandomSampler(training_indexes)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indexes)

    training_loader = torch.utils.data.DataLoader(
        dataset=ellipse_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=training_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=ellipse_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=test_sampler,
    )

    return training_loader, test_loader


class EllipseDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, transform=None):
        self._root_dir = os.path.abspath(dataset_root_dir)
        self._images = os.listdir(os.path.join(self._root_dir, 'pics'))
        self._metadata = os.listdir(os.path.join(self._root_dir, 'T'))
        self._transform = transform

        self.ELLIPSE_MATCH_PATTERN = r'radius={0}_radius2={0}_rotation={0}'.format(r'[\d.]+')

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        ellipse_name = re.search(self.ELLIPSE_MATCH_PATTERN, self._images[index])
        if ellipse_name is None:
            return None
        ellipse_name = ellipse_name.group()
        ellipse_image = skimage.io.imread(os.path.join(self._root_dir, 'pics', self._images[index]))
        ellipse_metadata_match = [
            metadata_entry for metadata_entry in self._metadata if re.search(ellipse_name, metadata_entry) is not None
        ]
        assert len(ellipse_metadata_match) == 1
        ellipse_metadata_strings = open(os.path.join(self._root_dir, 'T', ellipse_metadata_match[0])).readlines()
        ellipse_metadata = torch.Tensor([float(line) for line in ellipse_metadata_strings])
        print("Loading photo . . . ")
        print(ellipse_name)
        transform = torchvision.transforms.ToTensor()
        ellipse_image = transform(ellipse_image)

        return {
            'image_tensor' : ellipse_image,
            'metadata_tensor' : ellipse_metadata,
        }


if __name__ == "__main__":
    generate_training_set('.')