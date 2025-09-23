import os
import argparse
from model import load_model
from data import load_data, preprocessing, inverse_rotation_A, inverse_rotation_B, inverse_translation
from utils import relative_error, density_kernel, combine_pre_post, open_config
from plot import plot_scatter, plot_density, plot_pdf

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import gaussian_kde, mannwhitneyu

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDN model")
    parser.add_argument('--eps-scatter', action='store_true', help='Show epsilon scatter plots')
    parser.add_argument('--E-scatter', action='store_true', help='Show energy scatter plots')
    parser.add_argument('--correlation', action='store_true', help='Print correlation')
    parser.add_argument('--procrustes', action='store_true', help='Print Procrustes disparity')
    parser.add_argument('--plot-density', action='store_true', help='Plot density estimation')
    parser.add_argument('--marginals', action='store_true', help='Plot marginal distributions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test set')
    parser.add_argument('--pdf', action='store_true', help='Plot pdf estimation')
    parser.add_argument('--name', type=str, default='mdn', help='Name of the model')
    parser.add_argument('--data', type=str, default='collision_dataset.txt', help='Path to the dataset file')
    parser.add_argument('--save-figures', action='store_true', help='Save figures instead of showing them.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize directory
    if args.save_figures and not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    
    model_path = os.path.join(MODEL_DIR, f"{args.name}.h5")
    config_path = os.path.join(MODEL_DIR, f"{args.name}_config.json")
    
    config = open_config(config_path)
    
    # Load and preprocess data
    data = load_data(args.data)
    _, _, x_test, y_test = preprocessing(data, test_size=config.get("test_size"), include_b=config.get("include_b"))
    
    # Load model    
    model = load_model(model_path, config_path, x_test)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Combine x_test and y_test for CTC and MDN
    CTC_t = combine_pre_post(x_test[:,1], y_test[:,0])
    CTC_r = combine_pre_post(x_test[:,2], y_test[:,1])
    MDN_t = combine_pre_post(x_test[:,1], y_pred[:,0])
    MDN_r = combine_pre_post(x_test[:,2], y_pred[:,1])
            
    # Plot scatter plots for MDN and CTC (Figure 4.7)
    if args.eps_scatter:
        plt.figure(figsize=[6,6])

        # MDN eps_t
        plt.subplot(2,2,1)
        MDN_t_slope, MDN_t_intercept = plot_scatter(MDN_t[:,0], MDN_t[:,1], 
                            xlabel=r"$\varepsilon_{t}^{(p)}$", 
                            ylabel=r"$\varepsilon_{t}'^{(p)}$", 
                            xlim=(-4.8, 4.8), 
                            ylim=(-4.8, 4.8), 
                            title='MDN', 
                            return_params=True)
        # CTC eps_t
        plt.subplot(2,2,2)
        CTC_t_slope, CTC_t_intercept = plot_scatter(CTC_t[:,0], CTC_t[:,1],             
                            xlabel=r"$\varepsilon_{t}^{(p)}$", 
                            ylabel=r"$\varepsilon_{t}'^{(p)}$", 
                            xlim=(-4.8, 4.8), 
                            ylim=(-4.8, 4.8), 
                            title='CTC', 
                            return_params=True)
        # MDN eps_r
        plt.subplot(2,2,3)
        MDN_r_slope, MDN_r_intercept = plot_scatter(MDN_r[:,0], MDN_r[:,1], 
                            xlabel=r"$\varepsilon_{r}^{(p)}$", 
                            ylabel=r"$\varepsilon_{r}'^{(p)}$", 
                            xlim=(-8, 8), 
                            ylim=(-8, 8), 
                            return_params=True)
        # CTC eps_r
        plt.subplot(2,2,4)
        CTC_r_slope, CTC_r_intercept = plot_scatter(CTC_r[:,0], CTC_r[:,1], 
                            xlabel=r"$\varepsilon_{r}^{(p)}$", 
                            ylabel=r"$\varepsilon_{r}'^{(p)}$", 
                            xlim=(-8, 8), 
                            ylim=(-8, 8), 
                            return_params=True)

        plt.tight_layout()
        
        if args.save_figures:
            fig_name = args.name + '_eps_scatter.png'
            plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=300)
        else:   
            plt.show()
        
        # Print slopes of regression lines 
        print("\n Slopes of regression lines (intercept=0):")
        print(f"  eps_t: relative difference: {relative_error(CTC_t_slope, MDN_t_slope)*100:.2f}%")
        print(f"  eps_r: relative difference: {relative_error(CTC_r_slope, MDN_r_slope)*100:.2f}%\n")
        print("-" * 40)
    
    # Plot pdf for MDN and CTC (Figure 4.8)
    if args.pdf:
        plt.figure(figsize=[9,4])
        
        ## Translational energy
        # MDN
        plt.subplot(1,3,1)
        E_t, E_tp_MDN = inverse_translation(x_test, y_pred)
        plot_pdf(E_tp_MDN, label='MDN')
        # CTC
        E_t, E_tp_CTC = inverse_translation(x_test, y_test)
        plot_pdf(E_tp_CTC, label='CTC', 
                    legend=True,
                    xlabel=r"$E_{tr}'/k_b$ (K)",
                    ylabel=r"$f(E_{tr}'/k_b)$",
                    title='Translational energy distribution',
                    xlim=(0, 4000)
                    )
        ## Rotational energy A
        # MDN
        plt.subplot(1,3,2)
        E_rA, E_rAp_MDN = inverse_rotation_A(x_test, y_pred)
        plot_pdf(E_rAp_MDN, label='MDN')
        # CTC
        E_rA, E_rAp_CTC = inverse_rotation_A(x_test, y_test)
        plot_pdf(E_rAp_CTC, label='CTC',
                    legend=True,
                    xlabel=r"$E_{r,A}'/k_b$ (K)",
                    ylabel=r"$f(E_{r,A}'/kb)$",
                    title='Rotational energy A distribution',
                    xlim=(0, 3000)
                    )
        
        ## Rotational energy B
        plt.subplot(1,3,3)
        # MDN   
        E_rB, E_rBp_MDN = inverse_rotation_B(x_test, y_pred)
        plot_pdf(E_rBp_MDN, label='MDN')
        # CTC
        E_rB, E_rBp_CTC = inverse_rotation_B(x_test, y_test)
        plot_pdf(E_rBp_CTC, label='CTC',
                    legend=True,
                    xlabel=r"$E_{r,B}'/k_b$ (K)",
                    ylabel=r"$f(E_{r,B}'/k_b)$",
                    title='Rotational energy B distribution',
                    xlim=(0, 3000)
                    )

        plt.tight_layout()
        
        if args.save_figures:
            fig_name = args.name + '_pdf_comparison.png'
            plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=300)
        else:
            plt.show()
    
    # MDN vs CTC energy scatter plots 
    if args.E_scatter:
        plt.figure(figsize=[6,8])

        ## Translational energy 
        # MDN
        plt.subplot(3,2,1)
        E_t, E_tp = inverse_translation(x_test, y_pred)
        plot_scatter(E_t, E_tp, 
                     title='MDN', 
                     xlabel=r"$E_{tr}/k_b$ (K)", 
                     ylabel=r"$E_{tr}'/k_b$ (K)", 
                     xlim=(0, 6000), 
                     ylim=(0, 10000))
        # CTC
        plt.subplot(3,2,2)
        E_t, E_tp = inverse_translation(x_test, y_test)
        plot_scatter(E_t, E_tp, 
                     title='CTC', 
                     xlabel=r"$E_{tr}/k_b$ (K)", 
                     ylabel=r"$E_{tr}'/k_b$ (K)", 
                     xlim=(0, 6000), 
                     ylim=(0, 10000))

        ## Rotational energies A
        # MDN
        plt.subplot(3,2,3)
        E_rA, E_rAp = inverse_rotation_A(x_test, y_pred)
        plot_scatter(E_rA, E_rAp, 
                     xlabel=r"$E_{r,A}/k_b$ (K)", 
                     ylabel=r"$E_{r,A}'/k_b$ (K)", 
                     xlim=(0, 3000), 
                     ylim=(0, 6300))
        # CTC
        plt.subplot(3,2,4)
        E_rA, E_rAp = inverse_rotation_A(x_test, y_test)
        plot_scatter(E_rA, E_rAp, 
                     xlabel=r"$E_{r,A}/k_b$ (K)", 
                     ylabel=r"$E_{r,A}'/k_b$ (K)", 
                     xlim=(0, 3000), 
                     ylim=(0, 6300))
        
        ## Rotational energies B
        # MDN
        plt.subplot(3,2,5)
        E_rB, E_rBp = inverse_rotation_B(x_test, y_pred)
        plot_scatter(E_rB, E_rBp, 
                     xlabel=r"$E_{r,B}/k_b$ (K)", 
                     ylabel=r"$E_{r,B}'/k_b$ (K)", 
                     xlim=(0, 3000), 
                     ylim=(0, 5500))
        # CTC
        plt.subplot(3,2,6)
        E_rB, E_rBp = inverse_rotation_B(x_test, y_test)
        plot_scatter(E_rB, E_rBp, 
                     xlabel=r"$E_{r,B}/k_b$ (K)", 
                     ylabel=r"$E_{r,B}'/k_b$ (K)", 
                     xlim=(0, 3000), 
                     ylim=(0, 5500))

        plt.tight_layout()
        
        if args.save_figures:
            fig_name = args.name + '_E_scatter.png'
            plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=300)
        else:
            plt.show()
    
    # Calculate correlation coefficients
    if args.correlation:
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
    
    # Procrustes analysis
    if args.procrustes:
        _, _, disparity_t = procrustes(CTC_t, MDN_t)
        _, _, disparity_r = procrustes(CTC_r, MDN_r)

        # Compare spatial similarity of CTC and MDN scatter plots
        print("\n Procrustes analysis:")
        print(f"  eps_t disparity: {disparity_t}")
        print(f"  eps_r disparity: {disparity_r}\n")
        print("-" * 40)

    # Plot density estimations
    if args.plot_density:
        plt.figure(figsize=(6,6))

        ## Density distribution of eps_t
        # MDN
        plt.subplot(2,2,1)
        plot_density(MDN_t, 
                     xlim=(-4.8, 4.8), 
                     ylim=(-4.8, 4.8),
                     xlabel=r"$\varepsilon_{t}^{(p)}$", 
                     ylabel=r"$\varepsilon_{t}'^{(p)}$", 
                     title='MDN')
        # CTC
        plt.subplot(2,2,2)
        plot_density(CTC_t, 
                     xlim=(-4.8, 4.8), 
                     ylim=(-4.8, 4.8),
                     xlabel=r"$\varepsilon_{t}^{(p)}$", 
                     ylabel=r"$\varepsilon_{t}'^{(p)}$", 
                     title='CTC')

        ## Density distribution of eps_r
        # MDN
        plt.subplot(2,2,3)
        plot_density(MDN_r, 
                     xlim=(-8, 8), 
                     ylim=(-8, 8),
                     xlabel=r"$\varepsilon_{r}^{(p)}$", 
                     ylabel=r"$\varepsilon_{r}'^{(p)}$")
        # CTC
        plt.subplot(2,2,4)
        plot_density(CTC_r, 
                     xlim=(-8, 8), 
                     ylim=(-8, 8),
                     xlabel=r"$\varepsilon_{r}^{(p)}$", 
                     ylabel=r"$\varepsilon_{r}'^{(p)}$")

        plt.tight_layout()
        
        if args.save_figures:
            fig_name = args.name + '_eps_density.png'
            plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=300)
        else:
            plt.show()
    
    # Plot marginal distributions and perform statistical test
    if args.marginals:
        # Compute marginal distributions for posterior energies
        Z_MDN_t, _, _, _, _ = density_kernel(MDN_t)
        Z_CTC_t, xmin_t, xmax_t, _, _ = density_kernel(CTC_t)
        Z_MDN_r, _, _, _, _ = density_kernel(MDN_r)
        Z_CTC_r, xmin_r, xmax_r, _, _ = density_kernel(CTC_r)
        
        marginal_mdn_t = np.sum(Z_MDN_t, axis=0)
        marginal_ctc_t = np.sum(Z_CTC_t, axis=0)
        marginal_mdn_r = np.sum(Z_MDN_r, axis=0)
        marginal_ctc_r = np.sum(Z_CTC_r, axis=0)

        # Normalize the marginal distributions
        x_axis_t = np.linspace(xmin_t, xmax_t, Z_MDN_t.shape[1])
        x_axis_r = np.linspace(xmin_r, xmax_r, Z_MDN_r.shape[1])
        # marginal_mdn_t /= np.trapz(marginal_mdn_t, x_axis_t)
        # marginal_ctc_t /= np.trapz(marginal_ctc_t, x_axis_t)
        # marginal_mdn_r /= np.trapz(marginal_mdn_r, x_axis_r)
        # marginal_ctc_r /= np.trapz(marginal_ctc_r, x_axis_r)

        plt.figure(figsize=(6,6))

        plt.subplot(2,1,1)
        plt.plot(x_axis_t, marginal_mdn_t, label='MDN')
        plt.plot(x_axis_t, marginal_ctc_t, label='CTC')
        plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
        plt.ylabel(r"p($\varepsilon_{t}'^{(p)}$)")
        plt.xlim(xmin_t, xmax_t)
        plt.title('Marginal distributions')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(x_axis_r, marginal_mdn_r, label='MDN')
        plt.plot(x_axis_r, marginal_ctc_r, label='CTC')
        plt.xlabel(r"$\varepsilon_{r}^{(p)}$")
        plt.ylabel(r"p($\varepsilon_{r}'^{(p)}$)")
        plt.xlim(xmin_r, xmax_r)
        plt.legend()

        plt.tight_layout()
        
        if args.save_figures:
            fig_name = args.name + '_marginal_distributions.png'
            plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=300)
        else:
            plt.show()
        
        # Perform Mann-Whitney U test to compare distributions
        U_t, p_t = mannwhitneyu(marginal_mdn_t, marginal_ctc_t)
        U_r, p_r = mannwhitneyu(marginal_mdn_r, marginal_ctc_r)

        print("\n Mann-Whitney U test:")
        print(f"  U_t: {U_t}")
        print("  Distribution is significantly different" if p_t < 0.05 else "  No significant difference in distribution")
        print(f"  U_r: {U_r}")
        print("  Distribution is significantly different" if p_r < 0.05 else "  No significant difference in distribution\n")
        print("-" * 40)
            
    if args.evaluate:        
        print(f"\n NLL: {model.evaluate(x_test, y_test)}")

if __name__ == "__main__":
    main()