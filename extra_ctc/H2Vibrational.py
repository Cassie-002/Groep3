# collision_jax_optimized.py
import math
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import jit, lax, vmap
import random

# Enable double precision in JAX (MATLAB uses double precision by default)
jax.config.update("jax_enable_x64", True)

# ---------------------------
# Physical constants & settings
# ---------------------------
ncoll = 10

# Molecule database
molecule_params = {
    "H2": {"m_atom": 1.6738e-27, "bondLength": 0.74e-10, "sigma": 2.72e-10, "eps_K": 10.00},
    "N2": {"m_atom": 2.3250e-26, "bondLength": 1.10e-10, "sigma": 3.69e-10, "eps_K": 95.93},
    "O2": {"m_atom": 2.6567e-26, "bondLength": 1.21e-10, "sigma": 3.46e-10, "eps_K": 118.0}
}
# I think the molecule parameters are for Single site LJ Models while we use Two site. I changed H2, but we need to find source for the rest.
# https://ris.utwente.nl/ws/portalfiles/portal/420745930/barraco-et-al-2023-comparison-of-eight-classical-lennard-jones-based-h2-molecular-models-in-the-gas-phase-at.pdf

molecule = "H2"

#d_H2 should be variable!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# I should also change
m_H  = molecule_params[molecule]["m_atom"]           # hydrogen atom mass [kg]
m_H2 = 2.0*m_H
sigma_LJ = molecule_params[molecule]["sigma"]        # hydrogen-hydrogen LJ sigma [m]
kB = 1.38064852e-23
d_H2_init = molecule_params[molecule]["bondLength"]             # hydrogen-hydrogen bond length [m]
I_init = 0.5 * (d_H2_init**2) * m_H
epsilon = molecule_params[molecule]["eps_K"]  * kB         # hydrogen-hydrogen LJ well depth [J]

#D_e= 38287*1.60217663e-19
D_e=7.607e-19
#delta = 0.72e-10
delta = 0.52e-10

dt = 0.1e-15
tsim = 2e-12
nSteps = int(tsim/dt)

seed = 42
key = jax.random.PRNGKey(seed)

# ---------------------------
# Helper functions
# ---------------------------

@jit
def skew(w):
    wx, wy, wz = w
    return jnp.array([[0.0, -wz, wy],
                      [wz,  0.0, -wx],
                      [-wy, wx, 0.0]])

@jit
def getRandRotMat(key1, key2):
    psi = jax.random.uniform(key1, (), minval=0.0, maxval=2*jnp.pi)
    phi = jnp.arccos(1 - 2*jax.random.uniform(key2, ()))
    theta = 0.0

    Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                    [jnp.sin(psi),  jnp.cos(psi), 0],
                    [0,             0,            1]])
    Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                    [0, 1, 0],
                    [-jnp.sin(theta), 0, jnp.cos(theta)]])
    Rx = jnp.array([[1, 0, 0],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi),  jnp.cos(phi)]])
    return Rz @ Ry @ Rx

@jit
def LJ_e(r, sigma=sigma_LJ, eps=epsilon):
    sr6 = (sigma / r)**6
    return 4.0 * eps * (sr6*sr6 - sr6)

@jit
def LJ_force_scalar(r, sigma=sigma_LJ, eps=epsilon):
    s6 = sigma**6
    r7 = r**7
    r13 = r**13
    return 24.0*eps*(2.0*(s6**2)/r13 - s6/r7)

#https://www.researchgate.net/publication/351076224_A_Theoretical_Study_on_Vibrational_Energies_of_Molecular_Hydrogen_and_Its_Isotopes_Using_a_Semi-classical_Approximation
#D_e= 38287 [cm-1], delta = 0.72 [A]
@jit
def M_e(r):
    return D_e*(1-jnp.exp(-(r-d_H2_init)/delta))**2

@jit
def M_force_scalar(r):
    return -2*D_e*(jnp.exp(-(r-d_H2_init)/delta)-jnp.exp(-2*(r-d_H2_init)/delta))/delta

@jit
def r_Me(E):
    return d_H2_init-delta*jnp.log(1+random.choice([-1,1])*jnp.sqrt(E/D_e))

@jit
def r_Me2(E,r):
    return jnp.where(r>d_H2_init, d_H2_init-delta*jnp.log(1-jnp.sqrt(E/D_e)), d_H2_init-delta*jnp.log(1+jnp.sqrt(E/D_e)))

@jit
def getFij(Xi, Xj):
    rij = Xi - Xj
    r = jnp.linalg.norm(rij)
    fmag = LJ_force_scalar(r)
    return jnp.where(r>0, (fmag / r) * rij, jnp.zeros(3))

@jit 
def getFvib(Xi, Xj):
    rij = Xi - Xj
    r = jnp.linalg.norm(rij)
    fmag = M_force_scalar(r)
    return jnp.where(r>d_H2_init, -fmag, fmag)



@jit
def getM(F13tr, F14tr, F23tr, F24tr, R1, R2, dH2_1, dH2_2):
    F13_r = F13tr @ R1
    F14_r = F14tr @ R1
    F23_r = F23tr @ R1
    F24_r = F24tr @ R1
    
    F31_r = -F13tr @ R2
    F41_r = -F14tr @ R2
    F32_r = -F23tr @ R2
    F42_r = -F24tr @ R2

    # vibrational force should only be in z direction and F = -k dH/2 as a test. However we need to check direction. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    M1_x = -dH2_1/2 * (F13_r[1] + F14_r[1]) + dH2_1/2 * (F23_r[1] + F24_r[1])
    M1_y = dH2_1/2 * (F13_r[0] + F14_r[0]) - dH2_1/2 * (F23_r[0] + F24_r[0])
    M1_z = 0.0
    
    M2_x = -dH2_2/2 * (F31_r[1] + F32_r[1]) + dH2_2/2 * (F41_r[1] + F42_r[1])
    M2_y = dH2_2/2 * (F31_r[0] + F32_r[0]) - dH2_2/2 * (F41_r[0] + F42_r[0])
    M2_z = 0.0
    
    return jnp.array([M1_x, M1_y, M1_z]), jnp.array([M2_x, M2_y, M2_z])

@jit
def getVdot(F, m):
    return F / m

@jit
def getRdot(w, R):
    return R @ skew(w)

@jit
def getWdot(M_body, I_scalar):
    return M_body / I_scalar

@jit
def signed_sqrt(val, key, I):
    sign = jnp.where(jax.random.uniform(key) > 0.5, 1.0, -1.0)
    return sign * jnp.sqrt(2.0*val/I)

# ---------------------------
# Core simulation
# ---------------------------

@jit
def simulate_one_collision(keys):
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, kR11, kR12, kR21, kR22, w1_key, w2_key, w3_key, w4_key = keys

    #Generate vibrational energy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Random energies
    Etr_J = (100.0 + jax.random.uniform(k1)*5900.0) * kB
    vtr = jnp.sqrt(Etr_J / m_H2)
    b = jax.random.uniform(k2)*1.5*sigma_LJ

    Erot_tot_1 = jax.random.uniform(k3)*3000*kB
    Erot_tot_2 = jax.random.uniform(k4)*3000*kB
    frac11 = jax.random.uniform(k5)
    frac21 = jax.random.uniform(k6)
    Er11 = frac11*Erot_tot_1
    Er12 = (1.0-frac11)*Erot_tot_1
    Er21 = frac21*Erot_tot_2
    Er22 = (1.0-frac21)*Erot_tot_2

    Evib1 = jax.random.uniform(k7)*1500*kB
    fracvib1 = jax.random.uniform(k8)
    Epot1 = fracvib1*Evib1
    Ekin1 = (1.0-fracvib1)*Evib1

    Evib2 = jax.random.uniform(k9)*1500*kB
    fracvib2 = jax.random.uniform(k10)
    Epot2 = fracvib2*Evib2
    Ekin2 = (1.0-fracvib2)*Evib2

    E=Etr_J+Erot_tot_1+Erot_tot_2+Evib1+Evib2

    #Calculate this from energy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # atomic seperation
    d_H2_1 = r_Me(Epot1)
    I1 = 0.5 * (d_H2_1**2) * m_H
    d_H2_2 = r_Me(Epot2)
    I2 = 0.5 * (d_H2_2**2) * m_H

    # Angular velocities
    w11 = signed_sqrt(Er11, w1_key, I1)
    w12 = signed_sqrt(Er12, w2_key, I1)
    w21 = signed_sqrt(Er21, w3_key, I2)
    w22 = signed_sqrt(Er22, w4_key, I2)
    w1 = jnp.array([w11, w12, 0.0])
    w2 = jnp.array([w21, w22, 0.0])

    # Initial positions
    X1 = jnp.array([-2.0*sigma_LJ, 0.0, -b/2.0])
    X2 = jnp.array([2.0*sigma_LJ, 0.0, b/2.0])
    X11_0 = jnp.array([0.0,0.0,0.5*d_H2_1])
    X12_0 = jnp.array([0.0,0.0,-0.5*d_H2_1])
    X21_0 = jnp.array([0.0,0.0,0.5*d_H2_2])
    X22_0 = jnp.array([0.0,0.0,-0.5*d_H2_2])

    # Random rotations
    R1 = getRandRotMat(kR11, kR12)
    R2 = getRandRotMat(kR21, kR22)

    Xv11 = R1 @ X11_0.T 
    Xv12 = R1 @ X12_0.T
    Xv21 = R2 @ X21_0.T 
    Xv22 = R2 @ X22_0.T

    X11 = X1 + Xv11.T 
    X12 = X1 + Xv12.T
    X21 = X2 + Xv21.T
    X22 = X2 + Xv22.T

    #We can always start at maximum strech, then we don't have starting velocity, but that might bias sampling. Vibrational energy is divided in its own translational and potential energy. 
    #Its probably best to generate those fractions, and use the rotational frames to generate velocities in the common frame. (remember energy fractions can indicate 2 velocity directions.)
    #This starting velocity is on the molecule so we need a seperate for atom velocities, though they will always be along the axis of the molecule. Maybe we dont acctually need to translate from rotating frame. 
    V1 = jnp.array([vtr, 0.0, 0.0])
    V2 = jnp.array([-vtr,0.0,0.0])

    vibV1 = random.choice([-1,1])*jnp.sqrt(Ekin1 / m_H2)
    vibV2 = random.choice([-1,1])*jnp.sqrt(Ekin2 / m_H2)

    m1 = 2*m_H; m2 = 2*m_H

    # Helper for the while loop state
    state = (X1, X2, V1, V2, vibV1, vibV2, R1, R2, w1, w2, X11, X12, X21, X22, d_H2_1, d_H2_2, I1, I2, 0.0, 0)

    def cond_fn(state):
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, dr, step = state
        return (dr <= 5.0*sigma_LJ) & (step < nSteps)

    def body_fn(state):
        X1, X2, V1, V2, vibV1, vibV2, R1, R2, w1, w2, X11, X12, X21, X22, d_H2_1, d_H2_2, I1, I2, dr, step = state
        step += 1
        dr = jnp.linalg.norm(X1 - X2)

        # Include Forces from covalent bond. These forces work on atoms not molecules.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Forces
        F13 = getFij(X11, X21)
        F14 = getFij(X11, X22)
        F23 = getFij(X12, X21)
        F24 = getFij(X12, X22)
        F1 = F13 + F14 + F23 + F24
        F2 = -F1
        M1, M2 = getM(F13, F14, F23, F24, R1, R2, d_H2_1, d_H2_2)

        F12 = getFvib(X11, X12)
        F34 = getFvib(X21, X22)

        # Include steps for atom vibrations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Velocity Verlet
        V1_ = V1 + 0.5*dt*getVdot(F1,m1)
        V2_ = V2 + 0.5*dt*getVdot(F2,m2)
        vibV1_ = vibV1 + 0.5*dt*getVdot(F12,m1)
        vibV2_ = vibV2 + 0.5*dt*getVdot(F34,m2)
        X1_new = X1 + dt*V1_
        X2_new = X2 + dt*V2_
        d_H2_1_new = d_H2_1 + dt*vibV1_
        d_H2_2_new = d_H2_2 + dt*vibV2_
        X11_0_new = jnp.array([0.0,0.0,0.5*(d_H2_1_new)])
        X12_0_new = jnp.array([0.0,0.0,-0.5*(d_H2_1_new)])
        X21_0_new = jnp.array([0.0,0.0,0.5*(d_H2_2_new)])
        X22_0_new = jnp.array([0.0,0.0,-0.5*(d_H2_2_new)])
        R1_ = R1 + 0.5*dt*getRdot(w1,R1)
        R2_ = R2 + 0.5*dt*getRdot(w2,R2)
        w1_ = w1 + 0.5*dt*getWdot(M1,I1)
        w2_ = w2 + 0.5*dt*getWdot(M2,I2)

        # Full step update
        R1_new = R1 + dt*getRdot(w1_,R1_)
        R2_new = R2 + dt*getRdot(w2_,R2_)
        Xv11_new = R1_new @ X11_0_new.T
        Xv12_new = R1_new @ X12_0_new.T
        Xv21_new = R2_new @ X21_0_new.T
        Xv22_new = R2_new @ X22_0_new.T
        X11_new = X1_new + Xv11_new.T
        X12_new = X1_new + Xv12_new.T
        X21_new = X2_new + Xv21_new.T
        X22_new = X2_new + Xv22_new.T

        # Recompute forces at t+dt
        F13_new = getFij(X11_new, X21_new)
        F14_new = getFij(X11_new, X22_new)
        F23_new = getFij(X12_new, X21_new)
        F24_new = getFij(X12_new, X22_new)
        F1_new = F13_new + F14_new + F23_new + F24_new
        F2_new = -F1_new
        M1_new, M2_new = getM(F13_new,F14_new,F23_new,F24_new,R1_new,R2_new,d_H2_1, d_H2_2)

        F12_new = getFvib(X11_new, X12_new)
        F34_new = getFvib(X21_new, X22_new)
        I1_new = 0.5 * (d_H2_1_new**2) * m_H
        I2_new = 0.5 * (d_H2_2_new**2) * m_H


        V1_new = V1_ + 0.5*dt*getVdot(F1_new,m1)
        V2_new = V2_ + 0.5*dt*getVdot(F2_new,m2)
        vibV1_new = vibV1_ + 0.5*dt*getVdot(F12_new,m1)
        vibV2_new = vibV2_ + 0.5*dt*getVdot(F34_new,m2)
        w1_new = w1_ + 0.5*dt*getWdot(M1_new,I1_new)
        w2_new = w2_ + 0.5*dt*getWdot(M2_new,I2_new)

        #Include Evib !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Get new energies
        Etr_new =0.5*m1*jnp.linalg.norm(V1_new)**2 + 0.5*m2*jnp.linalg.norm(V2_new)**2
        Erot1_new =0.5*I1_new*(w1_new[0]**2 + w1_new[1]**2)
        Erot2_new =0.5*I2_new*(w2_new[0]**2 + w2_new[1]**2)

        Ekin1_new = 0.5*m1*vibV1_new**2
        Epot1_new = M_e(d_H2_1_new)
        Evib1_new = Ekin1_new +Epot1_new

        Ekin2_new = 0.5*m2*vibV2_new**2
        Epot2_new = M_e(d_H2_2_new)
        Evib2_new = Ekin2_new +Epot2_new

        #Normalization is fucked !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Normalize energy
        
        E_new = Etr_new + Erot1_new +Erot2_new+Evib1_new+Evib2_new
        norm = E/E_new
        Epot1_new = norm*Epot1_new
        Epot2_new = norm*Epot2_new
        d_H2_1_new = r_Me2(Epot1_new,d_H2_1_new)
        d_H2_2_new = r_Me2(Epot1_new,d_H2_2_new)

        norm = jnp.sqrt(norm)
        V1_new = norm*V1_new
        V2_new = norm*V2_new
        w1_new = norm*w1_new
        w2_new = norm*w2_new
        vibV1_new = norm*vibV1_new
        vibV2_new = norm*vibV2_new

        I1_new = 0.5 * (d_H2_1_new**2) * m_H
        I2_new = 0.5 * (d_H2_2_new**2) * m_H

     



        return (X1_new, X2_new, V1_new, V2_new, vibV1_new, vibV2_new, R1_new, R2_new, w1_new, w2_new, X11_new, X12_new, X21_new, X22_new,
                 d_H2_1_new, d_H2_2_new, I1_new, I2_new, dr, step)

    X1f, X2f, V1f, V2f, vibV1f, vibV2f, R1f, R2f, w1f, w2f, X11_new, X12_new, X21_new, X22_new, d_H2_1f, d_H2_2f, I1f, I2f, drf, _ = lax.while_loop(cond_fn, body_fn, state)

    # We need to calculate the energies from atom velocities and bondlength!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Make sure initial and final I!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Energies
    Etr_init = 0.5*m1*jnp.linalg.norm(jnp.array([vtr,0,0]))**2 + 0.5*m2*jnp.linalg.norm(jnp.array([-vtr,0,0]))**2
    Erot1_init = 0.5*I1*(w1[0]**2 + w1[1]**2)
    Erot2_init = 0.5*I2*(w2[0]**2 + w2[1]**2)

    Etr_final = 0.5*m1*jnp.linalg.norm(V1f)**2 + 0.5*m2*jnp.linalg.norm(V2f)**2
    Erot1_final = 0.5*I1f*(w1f[0]**2 + w1f[1]**2)
    Erot2_final = 0.5*I2f*(w2f[0]**2 + w2f[1]**2)
    Ekin1_final = 0.5*m1*vibV1f**2
    Epot1_final = M_e(d_H2_1f)
    Evib1_final = Ekin1_final +Epot1_final
    Ekin2_final = 0.5*m2*vibV2f**2
    Epot2_final = M_e(d_H2_2f)
    Evib2_final = Ekin2_final +Epot2_final

    return jnp.array([b/sigma_LJ]), jnp.array([Etr_init/kB]), jnp.array([Erot_tot_1/kB]), jnp.array([Erot_tot_2/kB]), jnp.array([Evib1/kB]), jnp.array([Evib2/kB]), jnp.array([Etr_final/kB]), jnp.array([Erot1_final/kB]), jnp.array([Erot2_final/kB]), jnp.array([Evib1_final/kB]), jnp.array([Evib2_final/kB]), jnp.array([d_H2_1f])

# ---------------------------
# Run collisions
# ---------------------------
keys_all = jax.random.split(key, ncoll*18).reshape(ncoll, 18, 2)

if ncoll == 1:
    results = simulate_one_collision(keys_all[0])
else:
    results = vmap(simulate_one_collision)(keys_all)

b_list, Etr_init_list, Er1_init_list, Er2_init_list, Ev1_init_list, Ev2_init_list, Etr_final_list, Er1_final_list, Er2_final_list, Ev1_final_list, Ev2_final_list, dtest = results


# Convert to numpy arrays for DataFrame
b_list = np.array(b_list).flatten()
Etr_init_list = np.array(Etr_init_list).flatten()
Er1_init_list = np.array(Er1_init_list).flatten()
Er2_init_list = np.array(Er2_init_list).flatten()
Ev1_init_list = np.array(Ev1_init_list).flatten()
Ev2_init_list = np.array(Ev2_init_list).flatten()
Etr_final_list = np.array(Etr_final_list).flatten()
Er1_final_list = np.array(Er1_final_list).flatten()
Er2_final_list = np.array(Er2_final_list).flatten()
Ev1_final_list = np.array(Ev1_final_list).flatten()
Ev2_final_list = np.array(Ev2_final_list).flatten()
dtest = np.array(dtest).flatten()
print(dtest)

df = pd.DataFrame({
    'b': np.array(b_list),
    'Etr': np.array(Etr_init_list),
    'Er1': np.array(Er1_init_list),
    'Er2': np.array(Er2_init_list),
    'Ev1': np.array(Ev1_init_list),
    'Ev2': np.array(Ev2_init_list),
    'Etrp': np.array(Etr_final_list),
    'Er1p': np.array(Er1_final_list),
    'Er2p': np.array(Er2_final_list),
    'Ev1p': np.array(Ev1_final_list),
    'Ev2p': np.array(Ev2_final_list),
    'dtest': np.array(dtest)
})

outname = f'collision_dataset2{ncoll}.csv'
df.to_csv(outname, index=False)
print(f"Saved dataset to {outname}")
print(getFvib(jnp.array([0,0,0]),jnp.array([0,0,1.0e-7])))
