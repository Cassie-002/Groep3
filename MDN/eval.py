import os
import argparse
from model import load_model
from data import load_data, preprocessing
from utils import dscatter

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDN model")
    parser.add_argument('--show-plot', action='store_true', help='Show prediction plot')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data = load_data('collision_dataset.txt')
    _, _, x_test, y_test = preprocessing(data, test_size=0.3)
    
    path = os.path.join(MODEL_DIR, 'mdn_weights.h5')
    model = load_model(path, x_test)
    
    pred = model.predict(x_test)
    
    if args.show_plot:
        plt.figure(figsize=[9,3])
        
        plt.subplot(1,2,1)
        eps_t, eps_tp, eps_t_dist = dscatter(x_test[:,1], y_test[:,0])
        plt.scatter(eps_t, eps_tp, c=eps_t_dist, s=10)
        plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
        
        plt.subplot(1,2,2)
        eps_r, eps_rp, eps_r_dist = dscatter(x_test[:,2], y_test[:,1])
        plt.scatter(eps_r, eps_rp, c=eps_r_dist, s=10)
        plt.xlabel(r"$\varepsilon_{r}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{r}'^{(p)}$")
        
        plt.show()
        
    print(f"NLL: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    main()