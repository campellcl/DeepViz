import os
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    """
    TinyImageNet: Custom torch.utils.data.Dataset for the Stanford CS231N (https://tiny-imagenet.herokuapp.com/)
     TinyImageNet dataset available here: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    URL: https://pytorch.org/docs/stable/data.html#module-torch.utils.data
    URL: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.train_root_dir = os.path.join(self.root_dir, 'train/')
        self.val_root_dir = os.path.join(self.root_dir, 'val/')
        self.test_root_dir = os.path.join(self.root_dir, 'test/')
        self.transform = transform
        self._len = None

    def __len__(self):
        if self._len is not None:
            return self._len

        num_samples = 0
        num_train_samples = 0
        num_val_samples = 0
        num_test_samples = 0

        # Get the number of samples in the training data:
        for class_label_dir in os.listdir(self.train_root_dir):
            num_sample_images_for_class = len(os.listdir(os.path.join(class_label_dir, 'images')))
            num_train_samples += num_sample_images_for_class
        num_samples += num_train_samples

        # Get the number of samples in the validation data:
        num_val_samples = len(os.listdir(os.path.join(self.val_root_dir, 'images')))
        num_samples += num_val_samples

        # Get the number of samples in the testing data:
        num_test_samples = len(os.listdir(os.path.join(self.test_root_dir, 'images')))
        num_samples += num_test_samples

        self._len = num_samples

    def __getitem__(self, item):
