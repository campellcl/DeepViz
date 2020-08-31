from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler
from PyTorch.Lib.EnumeratedTypes.DatasetTypes import DatasetTypes
from PyTorch.Lib.EnumeratedTypes.DatasetNames import DatasetNames
from PyTorch.Lib.UtilityMethods import show_image_batch

IS_DEBUG: bool


class CIFARTenDataLoaders:
    """
    CIFAR 10 dataset PyTorch DataLoader instances. Input data (PILImage objects of range [0, 1]) are converted to
     torch.Tensor objects (of range [-1, 1] in accordance with the "Standard Score/ z-score") and each of the three
     color channels (Red (R), Green (G), Blue (B)) are normalized. After normalization the mean of the data will be
     ~0 and the standard deviation: ~1.
    URL: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#loading-and-normalizing-cifar10
    URL: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7?u=laochanlam
    URL: https://en.wikipedia.org/wiki/Standard_score
    """

    def __init__(self, val_dataset_size_percentage: float, train_img_batch_size: int, val_img_batch_size: int,
                 test_img_batch_size: int, reshuffle_data_every_epoch: bool = False,
                 shuffle_dataset_during_partitioning: bool = True, data_loader_num_workers: int = 0,
                 is_debug: bool = True):
        """
        __init__:
        :param val_dataset_size_percentage: <float> What percentage of the training dataset should be allocated as a
         validation dataset. This value should range from [0, 1] inclusive.
        :param train_img_batch_size: <int> The number of images in a single training batch; e.g. the number of images
         the training DataLoader will provide on each iteration.
        :param val_img_batch_size: <int> The number of images in a single validation batch; e.g. the number of images
         the validation DataLoader will provide on each iteration.
        :param test_img_batch_size: <int> The number of images in a single testing batch; e.g. the number of images
         the testing DataLoader will provide on each iteration.
        :param reshuffle_data_every_epoch: <bool> If set to True the data belonging to the DataLoader instance will be
         reshuffled at every epoch (default: False).
         see: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        :param shuffle_dataset_during_partitioning: <bool> A boolean value indicating if the training dataset should
         be shuffled before being split into the validation dataset (in accordance with the provided percentage).
        :param data_loader_num_workers: <int> How many subprocesses to use for data loading. A value of 0 means that
         the data will be loaded in the main process.
         see: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        :param is_debug: <bool> A boolean flag indicating if debug information should be printed to the console during
         execution.
        """
        # Declare global variables:
        global IS_DEBUG
        # Initialize global variables:
        IS_DEBUG = is_debug
        self.val_dataset_size_percentage = val_dataset_size_percentage
        self.train_img_batch_size = train_img_batch_size
        self.val_img_batch_size = val_img_batch_size
        self.test_img_batch_size = test_img_batch_size
        self.reshuffle_data_every_epoch = reshuffle_data_every_epoch
        self.shuffle_dataset_during_partitioning = shuffle_dataset_during_partitioning
        self.data_loader_num_workers = data_loader_num_workers
        '''
        CIFAR 10 Summary Statistics computed by the script at PyTorch/Lib/Utils/DatasetSummaryStatistics.py
        '''
        self.dataset_image_channel_means = {
            'train': [0.4914, 0.4822, 0.4465]
        }
        self.dataset_image_channel_standard_deviations = {
            'train': [0.2112, 0.2086, 0.2121]
        }
        self.human_readable_class_labels = (
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )
        self.random_seed = 107
        self.train_data_loader, self.val_data_loader, self.test_data_loader = self.__get_data_loaders()

    def __get_data_loaders(self):
        """
        __get_data_loaders: This helper method instantiates and returns torch.utils.data.DataLoader instances for the
         training, validation, and testing of the CIFAR 10 torchvision.datasets object.
        :returns train_data_loader, val_data_loader, test_data_loader:
        :return train_data_loader: <torch.utils.data.DataLoader> A PyTest DataLoader instance which provides
         multi-process iterators, support for pinned-memory, etc. for the training dataset.
         See: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        :return val_data_loader: <torch.utils.data.DataLoader> A PyTest DataLoader instance which provides
         multi-process iterators, support for pinned-memory, etc. for the validation dataset.
         See: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        :return test_data_loader: <torch.utils.data.DataLoader> A PyTest DataLoader instance which provides
         multi-process iterators, support for pinned-memory, etc. for the testing dataset.
         See: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        """
        train_data_loader: DataLoader
        val_data_loader: DataLoader
        test_data_loader: DataLoader
        '''
        Note: transforms.Normalize does the following for each image channel in the input tensor:
            normalized_image_channel = (image_channel - image_channel_mean) / image_channel_standard_deviation
         Passing the values for (mean, std) of all 0.5 will normalize the image in the range of [-1, 1]
            A value of 0 will be converted to: -1 = (0 - 0.5)/0.5
            A value of .5 will be converted to: 0 = (0.5 - 0.5)/0.5
         To de-normalize we can do:
            de_normalized_image_channel = (normalized_image_channel * image_channel_standard_deviation) + image_chanel_mean
        '''
        training_transforms = transforms.Compose([
            transforms.ToTensor(),
            # These values are computed by the script in Utils/DatasetSummaryStatistics.py:
            transforms.Normalize(
                mean=self.dataset_image_channel_means['train'],
                std=self.dataset_image_channel_standard_deviations['train']
            )
        ])

        # Declare training dataset:
        if IS_DEBUG:
            print('Downloading training data...')
        train_dataset = CIFAR10(root='../../../Data/Datasets/CIFAR/10/', train=True, download=True, transform=training_transforms)

        # Declare validation dataset:
        if IS_DEBUG:
            print('Downloading validation data...')
        val_dataset = CIFAR10(root='../../../Data/Datasets/CIFAR/10/', train=True, download=True, transform=training_transforms)

        # Declare testing dataset:
        if IS_DEBUG:
            print('Downloading testing data...')
        test_dataset = CIFAR10(root='../../../Data/Datasets/CIFAR/10/', train=False, download=True, transform=training_transforms)

        # Partition training dataset into training and validation datasets:
        num_training_samples = len(train_dataset)
        training_dataset_indices = list(range(num_training_samples))
        center_index = int(np.floor(num_training_samples * self.val_dataset_size_percentage))
        if self.shuffle_dataset_during_partitioning:
            np.random.seed(self.random_seed)
            np.random.shuffle(training_dataset_indices)
        training_dataset_indices = training_dataset_indices[center_index:]
        validation_dataset_indices = training_dataset_indices[:center_index]

        train_sampler: Sampler
        val_sampler: Sampler

        if self.reshuffle_data_every_epoch:
            train_sampler = RandomSampler(data_source=training_dataset_indices, replacement=False)
            raise NotImplementedError('Still need to determine how to reshuffle a torch.utils.data.sampler.SubsetRandomSampler object.')
        else:
            # Samples elements randomly from the list of training indices without replacement (see
            #  https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler):
            train_sampler = SubsetRandomSampler(indices=training_dataset_indices)
            val_sampler = SubsetRandomSampler(indices=validation_dataset_indices)

        # Now create the final data loaders:
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_img_batch_size,
            shuffle=False,   # Shuffle the data every epoch. Each time you iterate the DataLoader a different sequence will appear than in the previous invocation
            sampler=train_sampler,
            num_workers=self.data_loader_num_workers,
            pin_memory=True
        )
        val_data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_img_batch_size,
            shuffle=False,   # Shuffle the data every epoch. Each time you iterate the DataLoader a different sequence will appear than in the previous invocation
            sampler=val_sampler,
            num_workers=self.data_loader_num_workers,
            pin_memory=True
        )
        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.test_img_batch_size,
            shuffle=False,  # Shuffle the data every epoch. Each time you iterate the DataLoader a different sequence will appear than in the previous invocation
            num_workers=self.data_loader_num_workers,
            pin_memory=True
        )
        return train_data_loader, val_data_loader, test_data_loader


# class TinyImageNetDataset(Dataset):
#     """
#     Tiny ImageNet 200 Dataset.
#     """
#
#     def __init__(self, root_dir: str = 'Data/TinyImageNet'):
#         self.root_dir = root_dir
#
#     def __getitem__(self, idx: int):
#         """
#         Fetches a data sample in the Tiny ImageNet dataset based on the provided index.
#         :param idx: <int> The specified index of the item to return.
#         :return:
#         """
#         raise NotImplementedError
#
#     def __len__(self):
#         raise NotImplementedError


if __name__ == '__main__':
    IS_DEBUG = True
    print('Will run in debug mode?: %s' % IS_DEBUG)

    if IS_DEBUG:
        print('Instantiating PyTorch DataLoaders...')
    cifar_data_loaders = CIFARTenDataLoaders(
        val_dataset_size_percentage=0.20,
        train_img_batch_size=10,
        val_img_batch_size=10,
        test_img_batch_size=10,
        reshuffle_data_every_epoch=False,
        shuffle_dataset_during_partitioning=True,
        data_loader_num_workers=1
    )
    print('Instantiated DataLoaders.')
    image_channel_means = Tensor(cifar_data_loaders.dataset_image_channel_means['train'])
    image_channel_stds = Tensor(cifar_data_loaders.dataset_image_channel_standard_deviations['train'])
    print('Displaying training image batch...')
    show_image_batch(
        dataset_name=DatasetNames.CIFAR_TEN,
        data_loader=cifar_data_loaders.train_data_loader,
        unormalized_image_channel_means=image_channel_means,
        unormalized_image_channel_standard_deviations=image_channel_stds,
        human_readable_class_labels=cifar_data_loaders.human_readable_class_labels
    )
