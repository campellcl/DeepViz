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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_image_channel_means = None
        self.dataset_image_channel_standard_deviations = None
        self.generate_summary_stats()

    def generate_summary_stats(self):
        if self.dataset_name == DatasetNames.CIFAR_TEN:
            self.__generate_cifar_ten_summary_stats()

    def generate_cifar_ten_summary_stats_in_memory(self):
        cifar_ten_dataset = CIFAR10(
            root=self.dataset_root_path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        # Use np.concatenate to stick all images together to form a 50,000 x 32 x 3
        print('woah')

    def __generate_cifar_ten_summary_stats(self):
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
        for batch_index, (image_batch, image_batch_targets) in enumerate(cifar_data_loader):
            image_batch = image_batch.to(device=self.device)
            # Rearrange batch from shape [B, C, W, H] to shape [B, C, W * H]:
            image_batch = image_batch.view(image_batch.size(0), image_batch.size(1), -1)
            # Maintain running count of number of samples:
            total_num_image_samples += image_batch.size(0)
            # Compute the running global sum of the mean and standard deviation of each image batch:
            mean_tensor += image_batch.mean(2).sum(0)
            var_tensor += image_batch.var(2).sum(0)

        # Now we take the average of the running sum of means:
        mean_tensor /= total_num_image_samples
        var_tensor /= total_num_image_samples
        std_tensor = torch.sqrt(var_tensor)
        self.dataset_image_channel_means = mean_tensor
        self.dataset_image_channel_standard_deviations = std_tensor


if __name__ == '__main__':
    cifar_summary_stats = DatasetSummaryStatistics(dataset_name=DatasetNames.CIFAR_TEN, dataset_root_path='DeepViz/Data/CIFAR/10/')
    print('CIFAR 10 Dataset Image Channel Means: %s' % cifar_summary_stats.dataset_image_channel_means)
    print('CIFAR 10 Dataset Image Channel Standard Deviations: %s' % cifar_summary_stats.dataset_image_channel_standard_deviations)