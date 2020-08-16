from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PyTorch.Lib.EnumeratedTypes.DatasetNames import DatasetNames


class DatasetSummaryStatistics:
    """
    GenerateSummaryStats: Generates the summary statistics (mean and standard deviation of each RGB image channel) for
     the provided dataset name.
    """

    def __init__(self, dataset_name: DatasetNames, dataset_root_path: str):
        self.dataset_name = dataset_name
        self.dataset_root_path = dataset_root_path

    def generate_summary_stats(self):
        summary_stats = None
        if self.dataset_name == DatasetNames.CIFAR_TEN:
            print('CIFAR 10')
            summary_stats = self.generate_cifar_ten_summary_stats()

    def generate_cifar_ten_summary_stats_in_memory(self):
        cifar_ten_dataset = CIFAR10(
            root=self.dataset_root_path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        # Use np.concatenate to stick all images together to form a 50,000 x 32 x 3
        print('woah')

    def generate_cifar_ten_summary_stats(self):
        # https://stackoverflow.com/a/60103056/3429090
        mean_tensor = None
        std_tensor = None

        # Note: here we convert the input torchvision PILImage object to a torch.Tensor object:
        cifar_ten_dataset = CIFAR10(
            root=self.dataset_root_path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        # Now we can get some useful information:
        num_sample_images = cifar_ten_dataset.data.shape[0]
        num_image_channels = cifar_ten_dataset.data.shape[-1]

        # Here we will calculate a running sum of image tensors size (32, 32) for each RGB channel: (3, 32, 32):
        image_channel_sums_tensor = torch.zeros(size=(3, 32, 32))

        # Initialize tensors to hold the mean and standard deviation of each image channel:
        mean_tensor = torch.zeros(num_image_channels)
        std_tensor = torch.zeros(num_image_channels)
        var_tensor = torch.zeros(num_image_channels)

        # Create the DataLoader object which will serve iterable batches during training:
        cifar_data_loader = DataLoader(
            dataset=cifar_ten_dataset,
            batch_size=20,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            timeout=0
        )
        # Alternate method of getting the number of samples in the Dataset associated with the DataLoader:
        # num_sample_images = len(cifar_data_loader.dataset)
        total_num_image_samples = 0
        # Compute the running mean and standard deviation calculations:
        for image_batch, image_batch_targets in cifar_data_loader:
            # Rearrange batch from shape [B, C, W, H] to shape [B, C, W * H]:
            image_batch = image_batch.view(image_batch.size(0), image_batch.size(1), -1)
            # Maintain running count of number of samples:
            total_num_image_samples += image_batch.size(0)
            # Compute the running global sum of the mean and standard deviation of each image batch:
            mean_tensor += image_batch.mean(2).sum(0)
            var_tensor += image_batch.var(2).sum(0)

            # sample_image = image_batch[0]
            # image_channel_sums_tensor += sample_image
            # sample_image_channel_means = sample_image.mean(dim=1).sum(dim=1)

            # Add up every channel across every image in the image batch:
            # image_batch_channel_sums = image_batch.sum(dim=0)
            # image_channel_sums_tensor += image_batch_channel_sums

            # image_batch_channel_means = image_batch.mean(dim=2).sum(dim=2)
            # mean_tensor += image_batch.
            # The last batch can have a smaller batch size if the dataset is not evenly divisible:
            # num_samples_in_batch = image_batch.size(0)
            # Convert the image_batch to
            # image_batch = image_batch.view(num_samples_in_batch, image_batch.size(1), -1)
        # Now we take the average of the running sum of means:
        mean_tensor /= total_num_image_samples
        var_tensor /= total_num_image_samples
        std_tensor = torch.sqrt(var_tensor)
        print('done!')
        '''
        We want an ouptut tensor of size (3, 1) which contains the average pixel value for each 32 x 32 RGB channel.
        Approach 1) sum up every 32 x 32 RGB image channel and divide by the total number of images.
        Approach 2) 
        '''
        # 1) Sum every channel and divide by total number of channels
        # 2) Take the average of each channel

if __name__ == '__main__':
    cifar_summary_stats = DatasetSummaryStatistics(dataset_name=DatasetNames.CIFAR_TEN, dataset_root_path='DeepViz/Data/CIFAR/10/')
    cifar_summary_stats.generate_summary_stats()