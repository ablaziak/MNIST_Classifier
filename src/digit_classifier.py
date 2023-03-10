import numpy as np

from src.digit_classification_models.digit_classification_models import models


class DigitClassifier:
    """
    A class for classifying digits using pre-trained models.

    Attributes:
    -----------
        classifier (object): An instance of the selected classification model.

    Methods:
    --------
        __init__(self, algorithm: str = "cnn"): Initializes a DigitClassifier object with
        the specified algorithm.
        classify(self, x: np.array) -> int: Returns the predicted digit for a given image array.
    """

    def __init__(self, algorithm: str = "cnn"):
        """
        Initializes a DigitClassifier object with the specified algorithm.

        Args:
        -----
            algorithm (str): The name of the classification algorithm to use. Defaults to "cnn".

        Raises:
        -------
            ValueError: If the specified algorithm is not available.
        """
        if algorithm in models:
            self.classifier = models.get(algorithm)()
        else:
            raise ValueError(
                f"Algorithm {algorithm} is not available. Choose from: {list(models.keys())}")
        self.classifier.load()

    def classify(self, x: np.array) -> int:
        """
        Returns the predicted digit for a given image array.

        Args:
        -----
            x (np.array): An array representing the image of a handwritten digit.
            Should have shape (28, 28, 1).

        Raises:
        -------
            ValueError: If the input array does not have shape (28, 28, 1).

        Returns:
        --------
            int: The predicted digit for the input image.
        """
        if x.shape != (28, 28, 1):
            raise ValueError("np.array with shape (28, 28, 1) expected")
        result = self.classifier.predict(x)
        return result
