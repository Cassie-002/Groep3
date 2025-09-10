import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import sigmoid, inv_sigmoid

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def load_data(file_name, path=DATA_DIR):
    file = os.path.join(path, file_name)    
    data = pd.read_csv(file)
    return data

def preprocessing(data, test_size=0.3):
    train, test = train_test_split(data, test_size=test_size)

    Ec_train, eps_t_train, eps_r1_train, eps_tp_train, eps_r1p_train = compute_features(train)
    Ec_test, eps_t_test, eps_r1_test, eps_tp_test, eps_r1p_test = compute_features(test)
        
    # Concatenate features
    x_train = np.vstack((np.log(Ec_train), inv_sigmoid(eps_t_train), inv_sigmoid(eps_r1_train))).T
    y_train = np.vstack((inv_sigmoid(eps_tp_train), inv_sigmoid(eps_r1p_train))).T
    x_test = np.vstack((np.log(Ec_test), inv_sigmoid(eps_t_test), inv_sigmoid(eps_r1_test))).T
    y_test = np.vstack((inv_sigmoid(eps_tp_test), inv_sigmoid(eps_r1p_test))).T
    
    return x_train, y_train, x_test, y_test

# Compute new (dimensionless) energies
def compute_features(df):
    Ec = total_energy(df, ['Etr', 'Er1', 'Er2'])
    Ecp = total_energy(df, ['Etrp', 'Er1p', 'Er2p'])
    
    eps_t = ratio_translational(df, 'Etr', Ec)
    eps_tp = ratio_translational(df, 'Etrp', Ecp)
    
    eps_r1 = ratio_rotational(df, 'Er1', 'Er2')
    eps_r1p = ratio_rotational(df, 'Er1p', 'Er2p')
    
    return Ec, eps_t, eps_r1, eps_tp, eps_r1p

# E_c = E_tr + E_r1 + E_r2
def total_energy(df, cols):
    return df[cols].sum(axis=1).values

# epsilon_t = E_tr / E_c
def ratio_translational(df, col, tot_energy):
    return df[col].values / tot_energy

# epsilon_r1 = E_r1 / (E_r1 + E_r2)
def ratio_rotational(df, col1, col2):
    return df[col1].values / (df[col1].values + df[col2].values)