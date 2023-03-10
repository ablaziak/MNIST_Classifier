from abc import abstractmethod


class DigitClassificationInterface:
    """
    An interface for digit classification models.

    Methods:
    ________
        __build(): Builds the model architecture.
        train(self): Trains the model using training data.
        predict(self, x): Predicts the digit for a given image array.
        load(self, path): Loads a saved model.
        save(self, path): Saves the current model to disk.
    """

    @staticmethod
    @abstractmethod
    def __build():
        """
        Abstract method to build the model architecture.

        Raises:
        -------
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model using training data.

        Raises:
        -------
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x) -> int:
        """
        Abstract method to predict the digit for a given image array.

        Args:
        -----
            x (numpy.ndarray): The image array to be predicted.

        Returns:
        --------
            int: The predicted digit.

        Raises:
        -------
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
        """
        Abstract method to load a saved model.

        Args:
        -----
            path (str): The path to the saved model file.

        Returns:
        --------
            object: The loaded model object.

        Raises:
        -------
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        """
        Abstract method to save the current model to disk.

        Args:
        -----
            path (str): The path to save the model file.

        Raises:
        -------
            NotImplementedError: This method should be implemented in the derived class.
        """
        raise NotImplementedError()
