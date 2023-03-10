import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential

from src.data.data_processor import DataProcessor
from src.digit_classification_models.digit_classification_interface import \
    DigitClassificationInterface


class DigitClassificationRF(DigitClassificationInterface):
    """
    A class for digit classification using a random forest model.

    Attributes:
    -----------
        model (sklearn.ensemble.RandomForestClassifier): A random forest model used for
        digit classification.

    Methods:
    --------
        __init__(self): Initializes a DigitClassificationRF object.
        __build(): Builds a random forest model.
        train(self): Trains the random forest model.
        predict(self, x): Predicts the digit for a given image array.
        load(self, path="models/rf.h5"): Loads a saved random forest model.
        save(self, path="models/rf.h5"): Saves the current random forest model to disk.
    """
    def __init__(self):
        """
        Initializes a DigitClassificationRF object.
        If a saved random forest model exists, it loads it,
        otherwise it creates a new one and trains it.
        """
        self.model: Sequential = self.load()
        if self.model is None:
            self.model = self.__build()
            self.train()

    @staticmethod
    def __build() -> RandomForestClassifier:
        """
        Builds a random forest model.

        Returns:
        ________
            model (sklearn.ensemble.RandomForestClassifier): A random forest model used for
            digit classification.
        """
        model = RandomForestClassifier(verbose=1)
        return model

    def train(self) -> None:
        """
        Trains and saves the random forest model.
        """
        data_processor = DataProcessor()
        x_train = data_processor.get_x_train()
        x_train = x_train.reshape(x_train.shape[0], 784)
        self.model.fit(x_train, data_processor.get_y_train())
        self.save()

    def predict(self, x) -> int:
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
        result = self.model.predict(x.reshape(1, 784))[0]
        digit = np.argmax(result)
        return digit

    def load(self, path="models/rf.h5") -> Optional[RandomForestClassifier, None]:
        """
        Loads random forest model from the given path.

        Args:
        -----
            path (str): A path to the saved model file.

        Returns:
        --------
            Optional[RandomForestClassifier, None]: Trained RandomForestClassifier model if path
            was correct, None otherwise
        """
        try:
            return pickle.load(open(path, 'rb'))
        except OSError:
            return None

    def save(self, path="models/rf.h5") -> None:
        """
        Saves the current random forest model to disk.

        Args:
        -----
            path (str): The path to save the model.
        """
        pickle.dump(self.model, open(path, 'wb'))
