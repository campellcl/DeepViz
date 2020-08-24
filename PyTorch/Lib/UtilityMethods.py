import matplotlib.pyplot as plt
# Typing imports:
from PyTorch.Lib.EnumeratedTypes.DatasetNames import DatasetNames
from PyTorch.Lib.EnumeratedTypes.DatasetTypes import DatasetTypes
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8?u=campellcl
    Author: Danylo Ulianych
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(data=mean)
        std = torch.as_tensor(data=std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: Tensor):
        return super().__call__(tensor=tensor.clone())


def denormalize_image_tensor(image_tensor: Tensor, image_channel_means: Tensor, image_channel_standard_deviations: Tensor):
    """

    :param tensor:
    :param image_channel_means:
    :param image_channel_standard_deviations:
    :return:
    """
    original_image_tensor: Tensor
    '''
    The Normalization step: 
        normalized_image_channel = (original_image_channel - original_image_channel_mean) / original_image_channel_standard_deviation
     In Code: 
        norm_img = image_tensor.sub_(img_channel_means).div_(image_channel_standard_deviations)
    Can be reverted with:
        original_image_channel = (normalized_image_channel * original_image_channel_standard_deviation) + original_image_channel_mean
    '''
    original_image_tensor = image_tensor.mul_(image_channel_standard_deviations).add_(image_channel_means)
    return original_image_tensor


def show_image_batch(dataset_name: DatasetNames, data_loader: DataLoader,
                     unormalized_image_channel_means: Tensor, unormalized_image_channel_standard_deviations: Tensor, human_readable_class_labels: tuple):
    # Set the visualization batch size on the DataLoader:

    # Determine how to de-normalize the input tensor:
    # denormalization_values = {'mean': None, 'std': None}
    # if dataset_name == DatasetNames.CIFAR_TEN:
    #     denormalization_values['mean'] = ()
    #     denormalization_values['std'] = ()

    # Get random images from the provided dataloader (of the specified batch size):
    image_batch_iterable = iter(data_loader)
    image_batch, labels = image_batch_iterable.next()

    # image = image_batch[0]
    # image_red = image[0]
    # image_green = image[1]
    # image_blue = image[2]

    # De-normalize every image in the image_batch:
    for i, img in enumerate(image_batch):
        for img_channel, img_channel_mean, img_channel_std in zip(img, unormalized_image_channel_means, unormalized_image_channel_standard_deviations):
            img_channel.mul_(img_channel_std).add_(img_channel_mean)

    def imshow(img_tensor):
        np_img = img_tensor.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    str_labels = [human_readable_class_labels[i] for i in labels]
    imshow(torchvision.utils.make_grid(tensor=image_batch, nrow=len(image_batch), normalize=False))


    print(' '.join('%5s' % [human_readable_class_labels[i] for i in labels]))
    # image_means = unormalized_image_channel_means.expand(size=(1, -1))
    # image_stds = unormalized_image_channel_standard_deviations.expand(size=(1, -1))
    # denormalized_image = image.mul(image_stds.transpose(0, 1)).add(image_means)

    # De-normalize the images:
    # Convert to numpy nd-array:
    # X = image_batch.numpy()
    #

    # Show images:



def imshow_tensor(image_tensor: Tensor, denormalization_values: dict):
    """

    :param image_tensor:
    :param denormalization_values: <dict> A dictionary of the form:
     denormalization_values = {'mean': (0, 0, 0)
    :return:
    """


def show_cifar_image(dataset_type: DatasetTypes):

    def _imshow_tensor(normalized_tensor: Tensor):
        # Un-normalize the input tensor:
        img = normalized_tensor / 2 + 0.5



