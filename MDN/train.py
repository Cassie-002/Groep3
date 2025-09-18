import os
import argparse
from data import load_data, preprocessing
from model import build_model
from plot import plot_loss
from utils import save_config, open_config

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    parser.add_argument('--include-b', action='store_true', help='Include impact parameter in input features')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level during training')
    parser.add_argument('--save-best-only', action='store_true', help='Save only the best model weights')
    parser.add_argument('--name', type=str, default='mdn', help='Name of the model')
    parser.add_argument('--data', type=str, default='collision_dataset.txt', help='Path to the dataset file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data = load_data(args.data)
    x_train, y_train, x_test, y_test = preprocessing(data, test_size=args.test_size, include_b=args.include_b)

    CB = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=args.verbose,
            restore_best_weights=True
        )
    ]

    mdn = build_model(args.nr_gaussians, args.activation_function, args.nr_neurons)
    
    history = mdn.fit(x_train, 
                        y_train, 
                        epochs=args.epochs, 
                        verbose=args.verbose,
                        validation_data=(x_test, y_test),
                        callbacks=CB)

    if args.show_loss:
        plot_loss(history)
    
    if args.save_model or args.save_best_only:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        model_path = os.path.join(MODEL_DIR, f"{args.name}.h5")
        config_path = os.path.join(MODEL_DIR, f"{args.name}_config.json")

        score = mdn.evaluate(x_test, y_test, verbose=0)
        config = {
                "nr_gaussians": args.nr_gaussians,
                "activation_function": args.activation_function,
                "nr_neurons": args.nr_neurons,
                "include_b": args.include_b,
                "epochs": args.epochs,
                "patience": args.patience,
                "data": args.data,
                "test_size": args.test_size,
                "final_loss": score
            }
               
        if args.save_best_only:
            if not os.path.exists(model_path):
                save_config(config, config_path)
                mdn.save_weights(model_path)
                print(f"No existing model found. Model weights saved to {model_path}")
            else:               
                old_config = open_config(config_path)
                old_score = old_config.get("final_loss")
                
                if score < old_score:
                    save_config(config, config_path)
                    mdn.save_weights(model_path)
                    print(f"New model improved from {old_score:.4f} to {score:.4f}. Saving new model.")
                else:
                    print(f"New model did not improve (old: {old_score:.4f}, new: {score:.4f}). Not saving.")
        else:
            save_config(config, config_path)
            mdn.save_weights(model_path)
            print(f"Model weights saved to {os.path.join(MODEL_DIR, 'mdn_weights.h5')}")

if __name__ == "__main__":
    main()