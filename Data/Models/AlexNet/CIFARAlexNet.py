import torch.nn as nn

# Number of classes in the CIFAR 10 dataset:
NUM_CLASSES: int = 10

class CIFARAlexNet(nn.Module):

    def __init__(self, num_classes_in_dataset):
        super(CIFARAlexNet, self).__init__()
        self.features = nn.Sequential(

        )