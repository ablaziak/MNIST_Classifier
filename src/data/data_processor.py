import numpy as np
from tensorflow.keras.utils import to_categorical


class DataProcessor:
    """
    Class that loads and preprocesses the MNIST dataset.

    Attributes:
    -----------
        mnist (numpy array): The MNIST dataset.

    Args:
    -----
        data_path (str): The path to the MNIST dataset file (default: 'data/mnist.npz').
    """

    def __init__(self, data_path='data/mnist.npz'):
        """
        Initializes a new DataProcessor instance.

        Args:
        -----
            data_path (str): The path to the MNIST dataset file (default: 'data/mnist.npz').
        """
        self.mnist = np.load(data_path)

    @staticmethod
    def preprocess_x(x):
        """
        Preprocesses the input images.

        Args:
        -----
            x (numpy array): The input images.

        Returns:
        --------
            numpy array: The preprocessed input images.
        """
        x = x.astype("float32") / 255.0
        x = x.reshape(x.shape[0], 28, 28, 1)
        return x

    @staticmethod
    def preprocess_y(y):
        """
        Preprocesses the input labels.

        Args:
        -----
            y (numpy array): The input labels.

        Returns:
        --------
            numpy array: The preprocessed input labels.
        """
        return to_categorical(y)

    def get_x_train(self):
        """
        Gets the preprocessed training images.

        Returns:
        --------
            numpy array: The preprocessed training images.
        """
        return self.preprocess_x(self.mnist['x_train'])

    def get_x_test(self):
        """
        Gets the preprocessed test images.

        Returns:
        --------
            numpy array: The preprocessed test images.
        """
        return self.preprocess_x(self.mnist['x_test'])

    def get_y_train(self):
        """
        Gets the preprocessed training labels.

        Returns:
        --------
            numpy array: The preprocessed training labels.
        """
        return self.preprocess_y(self.mnist['y_train'])

    def get_y_test(self):
        """
        Gets the preprocessed test labels.

        Returns:
        --------
            numpy array: The preprocessed test labels.
        """
        return self.preprocess_y(self.mnist['y_test'])
