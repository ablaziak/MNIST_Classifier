from src.digit_classification_models.digit_classification_cnn import DigitClassificationCNN
from src.digit_classification_models.digit_classification_random import DigitClassificationRandom
from src.digit_classification_models.digit_classification_random_forest import DigitClassificationRF

models = {
    "cnn": DigitClassificationCNN,
    "rf": DigitClassificationRF,
    "rand": DigitClassificationRandom
}
