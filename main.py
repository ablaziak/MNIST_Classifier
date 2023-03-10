from src.data.data_processor import DataProcessor
from src.digit_classifier import DigitClassifier

if __name__ == "__main__":
    data_processor = DataProcessor()
    x_test = data_processor.get_x_test()
    y_test = data_processor.get_y_test()

    digit_classifier = DigitClassifier(algorithm="rand")
    print(digit_classifier.classify(x_test[3]))
