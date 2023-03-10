import random

from src.digit_classification_models.digit_classification_interface import \
    DigitClassificationInterface


class DigitClassificationRandom(DigitClassificationInterface):
    """
    A class for digit classification using random guessing.

    Methods:
    ________
        __build(): A method to build a random guessing model (not applicable for this model).
        train(self): A method to train the model (not applicable for this model).
        predict(self, x): A method to predict a digit for a given image array using random guessing.
        load(self, path="models/rf.h5"): A method to load a saved model
        (not applicable for this model).
        save(self, path="models/rf.h5"): A method to save the current model to disk
        (not applicable for this model).
    """
    @staticmethod
    def __build():
        raise NotImplementedError("Not applicable for this model")

    def train(self):
        raise NotImplementedError("Not applicable for this model")

    def predict(self, x):
        """
        Predicts the digit for a given image array.

        Args:
        -----
            x (np.array): An array representing the image of a handwritten digit.
            Should have shape (28, 28, 1).

        Returns:
        --------
            int: The predicted digit for the input image.
        """
        return random.randint(0, 9)

    def load(self, path=""):
        raise NotImplementedError("Not applicable for this model")

    def save(self, path=""):
        raise NotImplementedError("Not applicable for this model")
