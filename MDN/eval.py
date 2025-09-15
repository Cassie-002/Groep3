import os
import argparse
from model import load_model
from data import load_data, preprocessing, inverse_rotation_A, inverse_rotation_B, inverse_translation
from utils import dscatter, regline, relative_error

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import gaussian_kde, mannwhitneyu

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDN model")
    parser.add_argument('--eps-scatter', action='store_true', help='Show epsilon scatter plots')
    parser.add_argument('--E-scatter', action='store_true', help='Show energy scatter plots')
    parser.add_argument('--include-b', action='store_true', help='Include impact parameter in input features')
    parser.add_argument('--correlation', action='store_true', help='Print correlation')
    parser.add_argument('--procrustes', action='store_true', help='Print Procrustes disparity')
    parser.add_argument('--plot-density', action='store_true', help='Plot density estimation')
    parser.add_argument('--marginals', action='store_true', help='Plot marginal distributions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test set')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load and preprocess data
    data = load_data('collision_dataset.txt')
    _, _, x_test, y_test = preprocessing(data, test_size=0.3, include_b=args.include_b)
    
    # Load model
    path = os.path.join(MODEL_DIR, 'mdn_weights.h5')
    model = load_model(path, x_test)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
     # Combine x_test and y_test for CTC and MDN
    CTC_t = np.vstack((x_test[:,1], y_test[:,0])).T
    MDN_t = np.vstack((x_test[:,1], y_pred[:,0])).T
    CTC_r = np.vstack((x_test[:,2], y_test[:,1])).T
    MDN_r = np.vstack((x_test[:,2], y_pred[:,1])).T
    
    # Plot scatter plots for MDN and CTC 
    if args.eps_scatter:
        plt.figure(figsize=[8,8])

        # MDN eps_t
        plt.subplot(2,2,1)
        eps_t, eps_tp, eps_t_mdn_dist = dscatter(MDN_t[:,0], MDN_t[:,1])
        x, y, MDN_t_slope, MDN_t_intercept = regline(MDN_t[:,0], MDN_t[:,1], intercept=False)
        plt.plot(x, y, 'r--')
        plt.scatter(eps_t, eps_tp, c=eps_t_mdn_dist, s=10)
        plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
        plt.xlim(-4.8, 4.8)
        plt.ylim(-4.8, 4.8)
        plt.title('MDN')

        # CTC eps_t
        plt.subplot(2,2,2)
        eps_t, eps_tp, eps_t_ctc_dist = dscatter(CTC_t[:,0], CTC_t[:,1])
        x, y, CTC_t_slope, CTC_t_intercept = regline(CTC_t[:,0], CTC_t[:,1], intercept=False)
        plt.plot(x, y, 'r--')
        plt.scatter(eps_t, eps_tp, c=eps_t_ctc_dist, s=10)
        plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
        plt.xlim(-4.8, 4.8)
        plt.ylim(-4.8, 4.8)
        plt.title('CTC')

        # MDN eps_r
        plt.subplot(2,2,3)
        eps_r, eps_rp, eps_r_mdn_dist = dscatter(MDN_r[:,0], MDN_r[:,1])
        x, y, MDN_r_slope, MDN_r_intercept = regline(MDN_r[:,0], MDN_r[:,1], intercept=False)
        plt.plot(x, y, 'r--')
        plt.scatter(eps_r, eps_rp, c=eps_r_mdn_dist, s=10)
        plt.xlabel(r"$\varepsilon_{r}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{r}'^{(p)}$")
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)

        # CTC eps_r
        plt.subplot(2,2,4)
        eps_r, eps_rp, eps_r_ctc_dist = dscatter(CTC_r[:,0], CTC_r[:,1])
        x, y, CTC_r_slope, CTC_r_intercept = regline(CTC_r[:,0], CTC_r[:,1], intercept=False)
        plt.plot(x, y, 'r--')
        plt.scatter(eps_r, eps_rp, c=eps_r_ctc_dist, s=10)
        plt.xlabel(r"$\varepsilon_{r}^{(p)}$")
        plt.ylabel(r"$\varepsilon_{r}'^{(p)}$")
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)

        plt.tight_layout()
        plt.show()
        
        # Print slopes of regression lines 
        print("\n Slopes of regression lines (intercept=0):")
        print(f"  eps_t: relative difference: {relative_error(CTC_t_slope, MDN_t_slope)*100:.2f}%")
        print(f"  eps_r: relative difference: {relative_error(CTC_r_slope, MDN_r_slope)*100:.2f}%\n")
        print("-" * 40)
    
    # MDN vs CTC energy scatter plots 
    if args.E_scatter:
        plt.figure(figsize=[8,9])

        ## Translational energy 
        plt.subplot(3,2,1)
        plt.title('MDN')
        plt.xlabel("Etr (K)")
        plt.ylabel("Etr' (K)")
        plt.xlim(0, 6000)
        plt.ylim(0, 10000)
        E_r, E_rp = inverse_translation(x_test, y_pred)
        E_r, E_rp, E_r_dist = dscatter(E_r, E_rp)
        plt.scatter(E_r, E_rp, c=E_r_dist, s=10)

        plt.subplot(3,2,2)
        plt.title('CTC')
        plt.xlabel("Etr (K)")
        plt.ylabel("Etr' (K)")
        plt.xlim(0, 6000)
        plt.ylim(0, 10000)
        E_r, E_rp = inverse_translation(x_test, y_test)
        E_r, E_rp, E_r_dist = dscatter(E_r, E_rp)
        plt.scatter(E_r, E_rp, c=E_r_dist, s=10)

        ## Rotational energies A
        plt.subplot(3,2,3)
        plt.xlabel("Er_A (K)")
        plt.ylabel("Er_A' (K)")
        plt.xlim(0, 3000)
        plt.ylim(0, 6300)
        E_rA, E_rAp = inverse_rotation_A(x_test, y_pred)
        E_rA, E_rAp, E_rA_dist = dscatter(E_rA, E_rAp)
        plt.scatter(E_rA, E_rAp, c=E_rA_dist, s=10)

        plt.subplot(3,2,4)
        plt.xlabel("Er_A (K)")
        plt.ylabel("Er_A' (K)")
        plt.xlim(0, 3000)
        plt.ylim(0, 6300)
        E_rA, E_rAp = inverse_rotation_A(x_test, y_test)
        E_rA, E_rAp, E_rA_dist = dscatter(E_rA, E_rAp)
        plt.scatter(E_rA, E_rAp, c=E_rA_dist, s=10)

        ## Rotational energies B
        plt.subplot(3,2,5)
        plt.xlabel("Er_B (K)")
        plt.ylabel("Er_B' (K)")
        plt.xlim(0, 3000)
        plt.ylim(0, 5500)
        E_rB, E_rBp = inverse_rotation_B(x_test, y_pred)
        E_rB, E_rBp, E_rB_dist = dscatter(E_rB, E_rBp)
        plt.scatter(E_rB, E_rBp, c=E_rB_dist, s=10)

        plt.subplot(3,2,6)
        plt.xlabel("Er_B (K)")
        plt.ylabel("Er_B' (K)")
        plt.xlim(0, 3000)
        plt.ylim(0, 5500)
        E_rB, E_rBp = inverse_rotation_B(x_test, y_test)
        E_rB, E_rBp, E_rB_dist = dscatter(E_rB, E_rBp)
        plt.scatter(E_rB, E_rBp, c=E_rB_dist, s=10)

        plt.tight_layout()
        plt.show()
    
    if args.correlation:
        # Calculate correlation coefficients
        CTC_t_corr = np.corrcoef(CTC_t[:,0], y=CTC_t[:,1])[0,1]
        MDN_t_corr = np.corrcoef(MDN_t[:,0], y=MDN_t[:,1])[0,1]
        CTC_r_corr = np.corrcoef(CTC_r[:,0], y=CTC_r[:,1])[0,1]
        MDN_r_corr = np.corrcoef(MDN_r[:,0], y=MDN_r[:,1])[0,1]
        
        print("\n Correlation coefficients:")
        print(f"  eps_t: CTC = {CTC_t_corr:.4f}, MDN = {MDN_t_corr:.4f}")
        print(f"    Relative difference: {relative_error(CTC_t_corr, MDN_t_corr)*100:.2f}%")
        print(f"  eps_r: CTC = {CTC_r_corr:.4f}, MDN = {MDN_r_corr:.4f}")
        print(f"    Relative difference: {relative_error(CTC_r_corr, MDN_r_corr)*100:.2f}%\n")
        print("-" * 40)
    
    if args.procrustes:
        _, _, disparity_t = procrustes(CTC_t, MDN_t)
        _, _, disparity_r = procrustes(CTC_r, MDN_r)

        # Compare spatial similarity of CTC and MDN scatter plots
        print("\n Procrustes analysis:")
        print(f"  eps_t disparity: {disparity_t}")
        print(f"  eps_r disparity: {disparity_r}\n")
        print("-" * 40)
    
    if args.evaluate:        
        print(f"\n NLL: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    main()