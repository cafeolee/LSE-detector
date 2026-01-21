from preprocessing_dataset import get_dataset
from train_classifier import train_model
from lse_detector import run_detector

def main():
    
    get_dataset()

    train_model()

    run_detector()


if __name__ == "__main__":
    main()