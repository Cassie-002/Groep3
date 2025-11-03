import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import jit, lax, vmap

# Enable double precision in JAX
jax.config.update("jax_enable_x64", True)

# ---------------------------
# Physical constants & settings
# ---------------------------
ncoll = 10000 #number of collisions
#The molecule parameters of Hydrogen are obtained from:
#https://ris.utwente.nl/ws/portalfiles/portal/420745930/barraco-et-al-2023-comparison-of-eight-classical-lennard-jones-based-h2-molecular-models-in-the-gas-phase-at.pdf
#The molecule parameters for nitrogen and oxygen are obtaind from:
#https://pdf.sciencedirectassets.com/271566/1-s2.0-S0022407300X03743/1-s2.0-002240739290142Q/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEvu8Tui437KNCCH83Jw1vYlg9e9hE8VDLmPahdB8VR%2BAiEA2qtobxtFYGbKWtQnaN2wV73inKN6P%2F7n7VP70b%2BGOgsquwUIo%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDMZzqmxd1jaNoWqyMiqPBXLbKJbcCQj%2FbfJFK2XvgwwPCW8LAGlO42iKnXzdfORv9vBFzkvvGneRS5Om2VKqZtXJUGPZfmhK7KudTQfQRnPQ4E2yflFKOQOZPnVpB9dCKqX3acD7g3YY6fSgU2ipx8Mu8WGAAmXiolilrpw0YbkxpD6PtRNLSKwSdByYc8CGB3nY7m9dNIt%2BHffnYRqFg%2B2TprAuG9BY%2FaHZp%2F%2FjfT21DgKfmn9h9PvZADpR9FuvQ6xC40hdKYWMTByzZLuau0FL8D29kA4kzbBSH4OMeVhpPdjWA0KTmbcsNuijO69hwDmfkPylIzz28CgSA6lO6GxPaBena9DsB5oQE6LmTMt45EOEh%2FGOd7EeugdSeN2%2F2dguLYNsQPnYeqUSH2zwe9dCYAZEIAKe6WLmN9f58wfg8hvXnGO0V9LXaqPxCVVDhsweozEa62FzSrSBDN4mUhRxfm6b4hc7aINZe7VFNju0ve7KdfDLHYYSxgVEf8b9zZijF1RE0IGGplrS%2BrzirTgx6CLc%2FKD2MMVGVUq8ouxUKIweC5jPjXiZyXznX7waGEqnt%2FCDg3Fwh54O4aQ30oJTliz7weBOqYrvqNfF2ymLefAojxb%2B25kHb%2B1Wog1TbWLMoPPxk3snqY1PAlJMMVDDNYdRgGCc4EOOnyUCccSNsasoabeJyIILZTjOo665OQCVX2o0GIkVTklMak5bT%2FCuD0VvBmFrYPpc6zZMuEeLW1bzVAH2C%2F0SOMmjqZFCm5C%2Bi30gMgy3pcDZecviExw9yj0VRTn7vo6QmVczcGAbWupcARpoiAoY6y6rBaXlMyXdQ8AfuE3rNsmz%2FgAmV6LgKg0hHeYNegPgfJvaU3YewATfvBnV0yVKnv8EhOsw0p%2FIxwY6sQHo5BkIRi5KBElWV8a%2BbmynyG5zWlTjcBdqRtokJyDLF1tBl29pzN%2Fid4AIiQPVDrqWtso5CXQ8fpT6mdv4XgLqPYMq7yzX4d7%2FDe3dZa02BOYcyL0yac%2BUvhNksb0b456BSJVMe%2FJOZjpPV910LVw4o5z389gGx%2Bm3ubA5%2FRGpledv2fgUUEcvLyzVaY0RtblPKjGeHrSNEM5S0C%2Bnrbft%2FzAuYeEqciXXj70aYmcPtfA%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251017T095603Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSUDCJPSX%2F20251017%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1a098ca9fa6a95eea95fc1b1acc4b70c2f705958cd6ba4e83e7b9b4e20fdee71&hash=24d9064ccb09a64e60d3f8433a413684048a331da12afa248ac3890d1e6c1052&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=002240739290142Q&tid=spdf-2037ae40-9e87-4717-b377-c0f114d042c4&sid=a89412d16d34d3449f1846c76e0ba989b007gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=140d595d010005025d53&rr=98feec838c870b5e&cc=nl
# Molecule database
molecule_params = {
    "H2": {"m_atom": 1.6738e-27, "bondLength": 0.74e-10, "sigma": 2.72e-10, "eps_K": 10.00},
    "N2": {"m_atom": 2.3250e-26, "bondLength": 1.09e-10, "sigma": 3.17e-10, "eps_K": 47.2},
    "O2": {"m_atom": 2.6567e-26, "bondLength": 1.21e-10, "sigma": 3.01e-10, "eps_K": 51.80}
}

#N2 "bondLength": 1.10e-10, "sigma": 3.29e-10, "eps_K": 37.2
#N2 "bondLength": 1.09e-10, "sigma": 3.17e-10, "eps_K": 47.2

molecule = "N2"

m_H  = molecule_params[molecule]["m_atom"]           # hydrogen atom mass [kg]
m_H2 = 2.0*m_H
sigma_LJ = molecule_params[molecule]["sigma"]        # hydrogen-hydrogen LJ sigma [m]
kB = 1.38064852e-23                                  # Boltzmann constant
d_H2 = molecule_params[molecule]["bondLength"]             # hydrogen-hydrogen bond length [m]
I = 0.5 * (d_H2**2) * m_H                                  # Moment of Inertia
epsilon = molecule_params[molecule]["eps_K"]  * kB         # hydrogen-hydrogen LJ well depth [J]

dt = 0.1e-16 #duration of a time step [s]
tsim = 2e-12 #maximum simulation duration [s]
nSteps = int(tsim/dt)

seed = 3141592654
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

#Lennard Jones potential
@jit
def LJ_e(r, sigma=sigma_LJ, eps=epsilon):
    sr6 = (sigma / r)**6
    return 4.0 * eps * (sr6*sr6 - sr6)

#Force from the LJ potential
@jit
def LJ_force_scalar(r, sigma=sigma_LJ, eps=epsilon):
    s6 = sigma**6
    r7 = r**7
    r13 = r**13
    return 24.0*eps*(2.0*(s6**2)/r13 - s6/r7)

#Calculates the force from Xj on Xi
@jit
def getFij(Xi, Xj):
    rij = Xi - Xj
    r = jnp.linalg.norm(rij)
    fmag = LJ_force_scalar(r)
    return jnp.where(r>0, (fmag / r) * rij, jnp.zeros(3))

#Torque
@jit
def getM(F13tr, F14tr, F23tr, F24tr, R1, R2, dH2):
    F13_r = F13tr @ R1
    F14_r = F14tr @ R1
    F23_r = F23tr @ R1
    F24_r = F24tr @ R1
    
    F31_r = -F13tr @ R2
    F41_r = -F14tr @ R2
    F32_r = -F23tr @ R2
    F42_r = -F24tr @ R2

    M1_x = -dH2/2 * (F13_r[1] + F14_r[1]) + dH2/2 * (F23_r[1] + F24_r[1])
    M1_y = dH2/2 * (F13_r[0] + F14_r[0]) - dH2/2 * (F23_r[0] + F24_r[0])
    M1_z = 0.0
    
    M2_x = -dH2/2 * (F31_r[1] + F32_r[1]) + dH2/2 * (F41_r[1] + F42_r[1])
    M2_y = dH2/2 * (F31_r[0] + F32_r[0]) - dH2/2 * (F41_r[0] + F42_r[0])
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

#returns either positive or negative root at random
@jit
def signed_sqrt(val, key):
    sign = jnp.where(jax.random.uniform(key) > 0.5, 1.0, -1.0)
    return sign * jnp.sqrt(2.0*val/I)

@jit
def pm_sqrt(val, key):
    sign = jnp.where(jax.random.uniform(key) > 0.5, 1.0, -1.0)
    return sign * jnp.sqrt(val)

# ---------------------------
# Core simulation
# ---------------------------

@jit
def simulate_one_collision(keys):
    kb, kx1, ky1, kz1, kx2, ky2, kz2, kvx1, kvy1,kvz1, kvx2, kvy2, kvz2, kR11, kR12, kR21, kR22, ktheta1, kphi1, ktheta2, kphi2, w1_key, w2_key, w3_key, w4_key = keys

    # Impact parameter
    b = jax.random.uniform(kb)*1.1*sigma_LJ
    
    # Random energies

    Emax = 1000.0
    T = 300

    # translational energy

    # Maxwell Boltzmann distrinution
    # particle 1
    Etrx1 = jax.random.chisquare(kx1,1) * T *kB/2
    Etry1 = jax.random.chisquare(ky1,1) * T *kB/2
    Etrz1 = jax.random.chisquare(kz1,1) * T *kB/2

    # particle 2
    Etrx2 = jax.random.chisquare(kx2,1) * T *kB/2
    Etry2 = jax.random.chisquare(ky2,1) * T *kB/2
    Etrz2 = jax.random.chisquare(kz2,1) * T *kB/2

    # Uniform distribution
    '''
    # particle 1
    Etrx1 = jax.random.uniform(kx1) * Emax *kB
    Etry1 = jax.random.uniform(ky1) * Emax *kB
    Etrz1 = jax.random.uniform(kz1) * Emax *kB

    # particle 2
    Etrx2 = jax.random.uniform(kx2) * Emax *kB
    Etry2 = jax.random.uniform(ky2) * Emax *kB
    Etrz2 = jax.random.uniform(kz2) * Emax *kB
    '''
    # velocity
    
    # particle 1
    vx1 = pm_sqrt(2* Etrx1 / m_H2, kvx1)
    vy1 = pm_sqrt(2* Etry1 / m_H2, kvy1)
    vz1 = pm_sqrt(2* Etrz1 / m_H2, kvz1)

    # particle 2
    vx2 = pm_sqrt(2* Etrx2 / m_H2, kvx2)
    vy2 = pm_sqrt(2* Etry2 / m_H2, kvy2)
    vz2 = pm_sqrt(2* Etrz2 / m_H2, kvz2)

    # relative velocity
    vtr = 0.5 * jnp.sqrt((vx2-vx1)**2 + (vy2-vy1)**2 + (vz2-vz1)**2)

    # Rotational energy

    # Maxwell Boltzmann distribution
    # particle 1
    Er11 = jax.random.chisquare(ktheta1,1) * T *kB/2
    Er12 = jax.random.chisquare(kphi1,1) * T *kB/2

    # particle 2
    Er21 = jax.random.chisquare(ktheta2,1) * T *kB/2
    Er22 = jax.random.chisquare(kphi2,1) * T *kB/2

    # Uniform distribution
    '''
    # particle 1
    Er11 = jax.random.uniform(ktheta1) * Emax *kB
    Er12 = jax.random.uniform(kphi1) * Emax *kB

    # particle 2
    Er21 = jax.random.uniform(ktheta2) * Emax *kB
    Er22 = jax.random.uniform(kphi2) * Emax *kB
    '''
    # Angular velocities
    w11 = signed_sqrt(Er11, w1_key)
    w12 = signed_sqrt(Er12, w2_key)
    w21 = signed_sqrt(Er21, w3_key)
    w22 = signed_sqrt(Er22, w4_key)
    w1 = jnp.array([w11, w12, 0.0])
    w2 = jnp.array([w21, w22, 0.0])

    # Initial positions
    X1 = jnp.array([-2.0*sigma_LJ, 0.0, -b/2.0])
    X2 = jnp.array([2.0*sigma_LJ, 0.0, b/2.0])
    X11_0 = jnp.array([0.0,0.0,0.5*d_H2])
    X12_0 = jnp.array([0.0,0.0,-0.5*d_H2])
    X21_0 = jnp.array([0.0,0.0,0.5*d_H2])
    X22_0 = jnp.array([0.0,0.0,-0.5*d_H2])

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

    V1 = jnp.array([vtr, 0.0, 0.0])
    V2 = jnp.array([-vtr,0.0,0.0])

    m1 = m_H2; m2 = m_H2

    # Helper for the while loop state
    state = (X1, X2, V1, V2, R1, R2, w1, w2, X11, X12, X21, X22, 0.0, 0)

    def cond_fn(state):
        _, _, _, _, _, _, _, _, _, _, _, _, dr, step = state
        return (dr <= 5.0*sigma_LJ) & (step < nSteps)

    def body_fn(state):
        X1, X2, V1, V2, R1, R2, w1, w2, X11, X12, X21, X22, dr, step = state
        step += 1
        dr = jnp.linalg.norm(X1 - X2)

        # Forces
        F13 = getFij(X11, X21)
        F14 = getFij(X11, X22)
        F23 = getFij(X12, X21)
        F24 = getFij(X12, X22)
        F1 = F13 + F14 + F23 + F24
        F2 = -F1
        M1, M2 = getM(F13, F14, F23, F24, R1, R2, d_H2)

        # Velocity Verlet
        V1_ = V1 + 0.5*dt*getVdot(F1,m1)
        V2_ = V2 + 0.5*dt*getVdot(F2,m2)
        X1_new = X1 + dt*V1_
        X2_new = X2 + dt*V2_
        R1_ = R1 + 0.5*dt*getRdot(w1,R1)
        R2_ = R2 + 0.5*dt*getRdot(w2,R2)
        w1_ = w1 + 0.5*dt*getWdot(M1,I)
        w2_ = w2 + 0.5*dt*getWdot(M2,I)

        # Full step update
        R1_new = R1 + dt*getRdot(w1_,R1_)
        R2_new = R2 + dt*getRdot(w2_,R2_)
        Xv11_new = R1_new @ X11_0.T
        Xv12_new = R1_new @ X12_0.T
        Xv21_new = R2_new @ X21_0.T
        Xv22_new = R2_new @ X22_0.T
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
        M1_new, M2_new = getM(F13_new,F14_new,F23_new,F24_new,R1_new,R2_new,d_H2)

        V1_new = V1_ + 0.5*dt*getVdot(F1_new,m1)
        V2_new = V2_ + 0.5*dt*getVdot(F2_new,m2)
        w1_new = w1_ + 0.5*dt*getWdot(M1_new,I)
        w2_new = w2_ + 0.5*dt*getWdot(M2_new,I)

        return (X1_new, X2_new, V1_new, V2_new, R1_new, R2_new, w1_new, w2_new,
                X11_new, X12_new, X21_new, X22_new, dr, step)

    X1f, X2f, V1f, V2f, R1f, R2f, w1f, w2f, _, _, _, _, drf, _ = lax.while_loop(cond_fn, body_fn, state)

    # Energies
    Etr_init = 0.5*m1*jnp.linalg.norm(jnp.array([vtr,0,0]))**2 + 0.5*m2*jnp.linalg.norm(jnp.array([-vtr,0,0]))**2
    Erot1_init = 0.5*I*(w1[0]**2 + w1[1]**2)
    Erot2_init = 0.5*I*(w2[0]**2 + w2[1]**2)
    E_init = Etr_init + Erot1_init +Erot2_init

    Etr_final = 0.5*m1*jnp.linalg.norm(V1f)**2 + 0.5*m2*jnp.linalg.norm(V2f)**2
    Erot1_final = 0.5*I*(w1f[0]**2 + w1f[1]**2)
    Erot2_final = 0.5*I*(w2f[0]**2 + w2f[1]**2)
    E_final = Etr_final + Erot1_final +Erot2_final


    return jnp.array([b/sigma_LJ]), jnp.array([Etr_init/kB]), jnp.array([Erot1_init/kB]), jnp.array([Erot2_init/kB]), jnp.array([Etr_final/kB]), jnp.array([Erot1_final/kB]), jnp.array([Erot2_final/kB])

# ---------------------------
# Run collisions
# ---------------------------
keys_all = jax.random.split(key, ncoll*25).reshape(ncoll, 25, 2)

if ncoll == 1:
    results = simulate_one_collision(keys_all[0])
else:
    results = vmap(simulate_one_collision)(keys_all)

b_list, Etr_init_list, Er1_init_list, Er2_init_list, Etr_final_list, Er1_final_list, Er2_final_list = results

# Convert to numpy arrays for DataFrame
b_list = np.array(b_list).flatten()
Etr_init_list = np.array(Etr_init_list).flatten()
Er1_init_list = np.array(Er1_init_list).flatten()
Er2_init_list = np.array(Er2_init_list).flatten()
Etr_final_list = np.array(Etr_final_list).flatten()
Er1_final_list = np.array(Er1_final_list).flatten()
Er2_final_list = np.array(Er2_final_list).flatten()

df = pd.DataFrame({
    'b': np.array(b_list),
    'Etr': np.array(Etr_init_list),
    'Er1': np.array(Er1_init_list),
    'Er2': np.array(Er2_init_list),
    'Etrp': np.array(Etr_final_list),
    'Er1p': np.array(Er1_final_list),
    'Er2p': np.array(Er2_final_list),
})

outname = f'collision_dataset_{molecule}_{ncoll}.csv'
df.to_csv(outname, index=False)
print(f"Saved dataset to {outname}")




