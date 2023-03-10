from typing import Optional

import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

from src.data.data_processor import DataProcessor
from src.digit_classification_models.digit_classification_interface import \
    DigitClassificationInterface


class DigitClassificationCNN(DigitClassificationInterface):
    """
    A class for digit classification using a convolutional neural network (CNN).

    Attributes:
    -----------
        model (tensorflow.python.keras.engine.sequential.Sequential): A sequential model used for
        digit classification.

    Methods:
    --------
        __init__(self): Initializes a DigitClassificationCNN object.
        __build(): Builds and compiles a CNN model.
        train(self): Trains the CNN model.
        predict(self, x): Predicts the digit for a given image array.
        load(self, path="models/cnn.h5"): Loads a saved CNN model.
        save(self, path="models/cnn.h5"): Saves the current CNN model to disk.
    """

    def __init__(self):
        """
        Initializes a DigitClassificationCNN object.
        If a saved CNN model exists, it loads it,
        otherwise it creates a new one and trains it.
        """
        self.model: Sequential = self.load()
        if self.model is None:
            self.model = self.__build()
            self.train()

    @staticmethod
    def __build() -> Sequential:
        """
        Builds and compiles a CNN model.

        Returns:
        --------
            tensorflow.python.keras.engine.sequential.Sequential: A compiled CNN model.
        """
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        model.compile(optimizer=RMSprop(lr=0.001),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    def train(self) -> None:
        """
        Trains and saves the CNN model.
        """
        data_processor = DataProcessor()
        self.model.fit(data_processor.get_x_train(),
                       data_processor.get_y_train(),
                       batch_size=128, epochs=30)
        self.save()

    def predict(self, x: np.array) -> int:
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
        result = self.model.predict(x.reshape(1, 28, 28, 1))
        digit = np.argmax(result)
        return digit

    def load(self, path: str = "models/cnn.h5") -> Optional[Sequential, None]:
        """
        Loads CNN model from the given path.

        Args:
        -----
            path (str): A path to the saved model file.

        Returns:
        --------
            Optional[Sequential, None]: Trained Sequential model if path was correct, None otherwise
        """
        try:
            return load_model(path)
        except OSError:
            return None

    def save(self, path: str = "models/cnn.h5") -> None:
        """
        Saves current CNN model to the given path.

        Args:
        -----
            path (str): The path to save the model.
        """
        self.model.save(path, include_optimizer=True)
