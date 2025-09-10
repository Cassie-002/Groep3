import os
import argparse
from data import load_data, preprocessing
from model import build_model
from utils import plot_loss

import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

def parse_args():
    parser = argparse.ArgumentParser(description="Train MDN model")
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--nr_gaussians', type=int, default=20, help='Number of Gaussians in MDN')
    parser.add_argument('--nr_neurons', type=int, default=8, help='Number of neurons in hidden layers')
    parser.add_argument('--activation_function', type=str, default='relu', help='Activation function for hidden layers')
    parser.add_argument('--show-loss', action='store_true', help='Show loss plot after training')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model weights')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proportion of test data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data = load_data('collision_dataset.txt')
    x_train, y_train, x_test, y_test = preprocessing(data, test_size=args.test_size)

    CB = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
            restore_best_weights=True
        )
    ]

    mdn = build_model(args.nr_gaussians, args.activation_function, args.nr_neurons)
    
    history = mdn.fit(x_train, 
                        y_train, 
                        epochs=args.epochs, 
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=CB)

    if args.show_loss:
        plot_loss(history)
    
    if args.save_model:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        mdn.save_weights(os.path.join(MODEL_DIR, 'mdn_weights.h5'))

if __name__ == "__main__":
    main()