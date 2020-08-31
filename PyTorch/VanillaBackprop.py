import os
import torch
from torch.autograd import Variable
from torch import Tensor
from torchvision import models
from numpy import ndarray
from PyTorch.Lib.DataLoaders.DataLoaders import CIFARTenDataLoaders

# Pre-trained model weights storage directory:
os.environ['TORCH_HOME'] = '../Data/Models'

global IS_GPU, IS_DEBUG


class VanillaBackprop:
    """
    VanillaBackprop: Performs standard/traditional backpropagation on the provided torchvision.models instance via
    """

    def __init__(self, model: models):
        """
        __init__: Initializer for objects of type VanillaBackprop.
        :param model: <torchvision.models> A torchvision.models instance representing the neural network architecture
         that should be utilized for training, and subsequently, visualization purposes.
        """
        self.model = model
        self.gradients = None
        '''
        We will leverage pre-trained models from torchvision.models. These models contain various modules
         which subclass torch.nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module. 
         Some nn.Modules (such as Dropout, and BatchNorm) function differently when put in eval() mode vs. train() mode
         (see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval). For the purposes
         of visualization, we do not want these modules active, so we will put the model in evaluation mode (see
         https://pytorch.org/docs/stable/torchvision/models.html). 
        '''
        self.model.eval()
        # We will attach functions/hooks to modules and layers in the pre-trained network:
        self._hook_layers()

    def _hook_layers(self):
        """
        _hook_layers: Attaches event emitters/function hooks to modules/layers in the pre-trained network which will
         fire during certain function calls. For instance, backward hooks fire during backpropagation (i.e.
         output.backward()) and forward hooks fire during forward propagation (i.e. output = model(input)). The best
         resource I can provide on understanding this stuff is: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
        :return:
        """

        def module_gradient_hook(module, grad_input, grad_output):
            """
            module_gradient_hook: This event emitter/function hook is executed during backpropagation and captures the
             gradients of the hooked module w.r.t. the input images. The captured gradients are stored in this wrapper
             classes' self.gradients. This hook will be executed when .backward() is called on the source model.
            :param module: <torch.nn.Module> The module (or layer) in the network for which the hook has been attached
             to; the implicit callee of this method.
            :param grad_input: <tuple> A 3-tuple containing the gradients entering the hooked module (in a backward
             direction) which were computed w.r.t the input of the layer via backpropagation. The three inputs represent
             each of the color channels in the source domain.
            :param grad_output: <tuple> A 1-tuple containing the gradients exiting the hooked module (in the backward
             direction). The output represents
            :return:
            """
            if IS_DEBUG:
                print('Now executing module_hook on module: %s' % module)
                print('Inside ' + self.model.__class__.__name__ + ' backward')
                print('Inside class: ' + self.model.__class__.__name__)
                print('')
                print('grad_input type: %s of size %d' % (type(grad_input), len(grad_input)))
                if grad_input[0] is not None:
                    print('\tgrad_input[0] type: ', type(grad_input[0]))
                    print('\tgrad_input[0] size: ', grad_input[0].size())
                else:
                    print('\tgrad_input[0] type: None')
                    print('\tgrad_input[0] size: None')
                if grad_input[1] is not None:
                    print('\tgrad_input[1] type: ', type(grad_input[1]))
                    print('\tgrad_input[1] size: ', grad_input[1].size())
                else:
                    print('\tgrad_input[1] type: None')
                    print('\tgrad_input[1] size: None')
                if grad_input[2] is not None:
                    print('\tgrad_input[2] type: ', type(grad_input[2]))
                    print('\tgrad_input[2] size: ', grad_input[2].size())
                else:
                    print('\tgrad_input[2] type: None')
                    print('\tgrad_input[2] size: None')
                print('')
                print('grad_output type: %s of size %d' % (type(grad_output), len(grad_output)))
                if grad_output[0] is not None:
                    print('\tgrad_output[0] type: ', type(grad_output[0]))
                    print('\tgrad_output[0]: ', grad_output[0].size())
                else:
                    print('\tgrad_output[0] type: None')
                    print('\tgrad_output[0] size: None')
            # Hook the first layer in the provided model to get the gradient:
            # see: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
            # see: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/23?u=campellcl
            '''
            "If you care about grad or outputs of some module, use module hooks, if you care about the grad w.r.t. some 
                Variable attach a hook to a Variable (there already is a register_hook function)".
            '''
            # Insert a module hook to capture the gradient of the very first layer during backpropagation:
            if self.model.features is not None:
                first_layer = self.model._modules['features'][0]
                first_layer.register_backward_hook(module_gradient_hook)
            else:
                network_layers_odict_keys = list(self.model._modules.keys())
                first_layer = self.model._modules.get(network_layers_odict_keys[0])
                first_layer.register_backward_hook(module_gradient_hook)

    def compute_gradients_for_single_image(self, input_image: Tensor, target_class_label: int):
        """
        compute_gradients_for_single_image: Computes the gradients for a single input image via one_hot_encoding and
         backpropagation.
        :param input_image: <torch.Tensor> A single sample image (read by a PyTorch DataLoader instance) for which the
         gradients are to be computed.
        :param target_class_label: <int> The target class label of the supplied image (in integer encoded non-human
         readable format).
        :return gradient_array <numpy.ndarray> A numpy array populated with the gradient of the input image, computed
         via backpropagation during a single training pass:
        """
        gradient_array: ndarray
        # Since there is only a single image, we need to add an extra preceding dimension to the Tensor in place of the minibatch size:
        # input_image = input_image.resize(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        input_image = input_image.view(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        # Now we compute a forward pass through the model for this single image:
        output = self.model(input_image)
        if IS_DEBUG:
            print('Model params: %s' % self.model.parameters)
            print('Output\'s grad_fn: %s' % output.grad_fn)
        final_layer_weights, preds = torch.max(output.data, 1)



if __name__ == '__main__':
    # Define global vars:
    IS_DEBUG = True
    print('Will run in debug mode?: %s' % IS_DEBUG)

    IS_GPU = torch.cuda.is_available()
    if IS_GPU:
        print('Detected CUDA compatible GPU. Operations will be performed on the GPU whenever practical.')
        device = torch.device("cuda:0")
    else:
        print('No CUDA compatible GPU detected. Operations will be performed solely on the CPU.')
        device = torch.device("cpu")

    cifar_data_loaders = CIFARTenDataLoaders(
        val_dataset_size_percentage=0.20,
        train_img_batch_size=10,
        val_img_batch_size=10,
        test_img_batch_size=10,
        reshuffle_data_every_epoch=False,
        shuffle_dataset_during_partitioning=True,
        data_loader_num_workers=1,
        is_debug=True
    )
    train_data_loader = cifar_data_loaders.train_data_loader
    val_data_loader = cifar_data_loaders.val_data_loader

    # Get random images from the provided dataloader (of the specified batch size):
    # image_batch_iterable = iter(train_data_loader)
    # train_image_batch, train_labels = image_batch_iterable.next()

    # Create the model used to visualize the data:
    alexnet = models.alexnet(pretrained=True)

    # Wrap this model in a custom class with utility methods for activation visualization:
    vanilla_backprop = VanillaBackprop(model=alexnet)

    # Iterate over every image batch in the training dataset:
    for image_batch_index, (images, labels) in enumerate(train_data_loader):
        # Wrap the image and labels in an Autograd variable so we can track all operations made to it (see
        # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html):
        if IS_GPU:
            images, labels = Variable(images.cuda(), requires_grad=True), Variable(labels.cuda(), requires_grad=False)
        else:
            images, labels = Variable(images, requires_grad=True), Variable(labels, requires_grad=False)

        # Iterate over every image in the current image batch:
        for image_index, (image, label) in enumerate(zip(images, labels)):
            gradient = vanilla_backprop.compute_gradients_for_single_image(input_image=image, target_class_label=label)


