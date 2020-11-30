import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
import matplotlib.pyplot as plt


is_debug: bool = True


def get_mnist_training_data_set(train_buffer_size: int = 60000, train_batch_size: int = 256):
    train_dataset: tf.data.Dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    num_train_images = train_images.shape[0]
    if is_debug:
        print('initial train_images.shape: %s' % (train_images.shape,))
        print('initial train_labels.shape: %s' % (train_labels.shape,))
        print('Number of training images: %s' % num_train_images)

    if is_debug:
        # MNIST images are 28 x 28 x 1 pixels (greyscale images):
        train_image = train_images[0]
        train_image_pixel = train_image[0][0]
        # The original type of the pixel data is numpy.uint8:
        print('Original Pixel data type: %s' % type(train_image_pixel))
        # Which means that the original pixel intensity ranges from 0 to ((2^8)-1) that is, [0 - 255] inclusive.
    mnist_image_height, mnist_image_width = 28, 28
    mnist_number_of_image_channels = 1
    # https://www.tensorflow.org/tutorials/generative/dcgan has us converting to a 32 bit float:
    train_images = train_images.reshape(num_train_images, mnist_image_height, mnist_image_width,
                                        mnist_number_of_image_channels).astype('float32')
    if is_debug:
        # The new type of the pixel data is numpy.float32:
        train_image = train_images[0]
        train_image_pixel = train_image[0][0][0]
        print('New Pixel data type: %s' % type(train_image_pixel))
    # The midpoint pixel intensity is used for normalization, this is half of the maximum pixel value (255):
    pixel_intensity_midpoint = 255 / 2  # 127.5
    # Normalize the images to the range [-1, 1] by subtracting and dividing by the midpoint pixel intensity:
    train_images = (train_images - pixel_intensity_midpoint) / pixel_intensity_midpoint
    # Batch and shuffle the training dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size=train_buffer_size)\
        .batch(batch_size=train_batch_size)
    return train_dataset


def make_dcgan_generator_model(kernel_size=7, input_image_length=28, input_image_width=28, input_image_resolution=256, num_image_channels=1, latent_vector_size=100):
    """
    make_generator_model: TODO: Docstring.
    The architecture for this tutorial is specified at the following URL:
     https://www.tensorflow.org/tutorials/generative/dcgan#create_the_models
    However, an actual explanation of this architecture is best found at this URL:
     https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#what-is-a-dcgan
     https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generator
    The default values for this method are sourced from the above URLs while adapting for the attributes of the MNIST
    dataset (as in the TensorFlow DCGAN tutorial). The addition of batch normalization functions after the dense fully
    connected layers is a major contribution fo the DCGAN paper. The strided Conv2DTranspose layers allow the latent
    vector to be transformed into a volume with the same shape as the input image.
    :param kernel_size: <int> The size of the kernel (i.e. feature maps) in the generator.
    :param input_image_resolution: <int> The pixel resolution of the input image. For an 8-bit image (e.g. MNIST) this
     value is 2^8=256.
    :param latent_vector_size: <int> This is the size of the latent 'z' vector (i.e. the size of the generator input).
    :return:
    """
    model = tf.keras.Sequential()
    ''' Project and reshape module: Project the latent z vector into an image-like shape: '''
    # Project and reshape 'z' as in https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generator:
    model.add(
        layers.Dense(kernel_size*kernel_size*input_image_resolution, use_bias=False, input_shape=(latent_vector_size,))
    )
    model.add(layers.BatchNormalization())
    # It appears that TensorFlow leverages LeakyRELU instead of RELU (which PyTorch uses) due to the current industry
    # bias toward non-saturating activation functions, but I cannot confirm this.
    # TODO: Why stick a LeakyReLU on the transformed and normalized latent vector?
    model.add(layers.ReLU())
    # TODO: Why the reshape here? What is the shape output of the LeakyReLU layer?
    model.add(layers.Reshape(target_shape=(kernel_size, kernel_size, input_image_resolution)))
    # Below the None says to ignore the batch_size for the assertion:
    assert model.output_shape == (None, kernel_size, kernel_size, input_image_resolution)   # default (?, 7, 7, 256)

    ''' CONV 1 module: Up-sample the random noise. '''
    model.add(
        layers.Conv2DTranspose(filters=int(input_image_resolution/2), kernel_size=(int(kernel_size*2), int(kernel_size*2)),
                               strides=(1, 1), padding='same', use_bias=False)
    )
    assert model.output_shape == (None, int(kernel_size*2), int(kernel_size*2), int(input_image_resolution/2)), model.output_shape    # default (?, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    ''' CONV 2 module: Up-sample again'''
    model.add(
        layers.Conv2DTranspose(filters=int(input_image_resolution/4), kernel_size=(kernel_size*4, kernel_size*4),
                               strides=(2, 2), padding='same', use_bias=False)
    )
    assert model.output_shape == (None, kernel_size*4, kernel_size*4, int(input_image_resolution/4))    # default (?, 28, 28, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    ''' CONV 3 module: Up-sample again '''
    model.add(layers.Conv2DTranspose(filters=int(input_image_resolution/8), kernel_size=(kernel_size*8, kernel_size*8),
                                     strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, kernel_size*8, kernel_size*8, int(input_image_resolution/8))    # default (?, 56, 56, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    ''' CONV 4 module: Output of generator (i.e. G(z)) '''
    # Here our final output is an MNIST image (batch_size, 1, 28, 28, 1)
    model.add(layers.Conv2DTranspose(filters=num_image_channels, kernel_size=(input_image_length, input_image_width),
                                     strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # model.add(layers.Activation(activation=activations.tanh))
    assert model.output_shape == (None, input_image_length, input_image_width, num_image_channels)
    return model



if __name__ == '__main__':
    is_debug = True
    print('Running tensorflow version: %s' % tf.__version__)
    train_dataset = get_mnist_training_data_set()

    # Use the untrained generator to create an image:
    generator = make_dcgan_generator_model()
    noise = tf.random.normal(shape=[1, 100], mean=0.0, stddev=1.0, seed=42)
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')