# collision_jax_hc_h2.py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, vmap
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

# FF ZIEN OF IE CPU OF GPU PAKT
print("JAX devices available:", jax.devices())
print("Default device:", jax.default_backend())


ncoll = 20000           
dt = 0.1e-15            # 0.1 fs
tsim = 2e-12            # 2 ps total 
nSteps = int(tsim / dt)

seed = 42
key = jax.random.PRNGKey(seed)

kB = 1.38064852e-23
NA = 6.02214076e23
T = 300.0               # K


# Molecule parameterS

m_H  = 1.6738e-27
m_H2 = 2.0 * m_H                                    # kg per H2 molecule
d_H2 = 0.741e-10                                    # bond length (m)
I_h2 = 0.5 * (d_H2**2) * m_H

# C9H20: paper gives reference diameter (d_ref) = 1.58 nm
m_C9H20 = 0.128 / NA                                # 128 g/mol -> 0.128 kg/mol -> per molecule. 
sigma_C9 = 1.58e-9                                  # m (paper's dref). 
eps_C9_K = 400.0                                    # assumed LJ epsilon in K (DEZE IS PLACEHOLDER)
epsilon_C9 = eps_C9_K * kB

# H2 LJ parameters 
sigma_H2 = 3.06e-10                                 
eps_H2_K = 34.0                                     # K 
epsilon_H2 = eps_H2_K * kB

# mix rules (DEZE KUNNEN WE GEWOON NOG UPDATEN, IS EEN PLACEHOLDER)
sigma_mix = 0.5 * (sigma_H2 + sigma_C9)
epsilon_mix = jnp.sqrt(epsilon_H2 * epsilon_C9)

# Hydrocarbon number density at POB entrance with H2 clean gas (Table 5): 1.46e13 m^-3
n_hc_target = 1.46e13                               # m^-3



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
def LJ_e(r, sigma=sigma_mix, eps=epsilon_mix):
    sr6 = (sigma / r)**6
    return 4.0 * eps * (sr6*sr6 - sr6)

@jit
def LJ_force_scalar(r, sigma=sigma_mix, eps=epsilon_mix):
    s6 = sigma**6
    r7 = r**7
    r13 = r**13
    return 24.0*eps*(2.0*(s6**2)/r13 - s6/r7)

@jit
def getFij(Xi, Xj):
    rij = Xi - Xj
    r = jnp.linalg.norm(rij)
    fmag = LJ_force_scalar(r)
    return jnp.where(r>0, (fmag / r) * rij, jnp.zeros(3))

@jit
def getVdot(F, m):
    return F / m

@jit
def getRdot(w, R):
    return R @ skew(w)

@jit
def getWdot(M_body, I_scalar):
    return M_body / I_scalar

# sample Maxwell speed magnitude for given mass and T
@jit
def sample_speed(key, m, T):
    # sample speed from Maxwell-Boltzmann distribution (magnitude)
    # using inverse transform for the Maxwell-Boltzmann speed distribution is cumbersome;
    # instead sample 3 normal components and return norm (simple and correct)
    v = jax.random.normal(key, (3,)) * jnp.sqrt(kB * T / m)
    return v, jnp.linalg.norm(v)

# Single collision 

@jit
def simulate_one_collision(keys):
    # keys: array of subkeys
    k_v1, k_v2, k_b, kR1, kR2, k_w1, k_w2 = keys

    # sample thermal velocities for H2 (as two-atom rigid body) and for C9H20 (single sphere)
    v_h2_vec, v_h2 = sample_speed(k_v1, m_H2, T)
    v_c9_vec, v_c9 = sample_speed(k_v2, m_C9H20, T)

    # set relative approach velocity along x axis scaled so collisions actually occur:
    # make them approach with relative speed ~ thermal relative speed
    v_rel = v_h2 - v_c9
    # ensure they are approaching: set H2 moving +x and C9H20 moving -x with relative v_rel magnitude
    V1 = jnp.array([+0.5 * v_rel, 0.0, 0.0]) + v_h2_vec - jnp.array([v_h2,0,0])
    V2 = jnp.array([-0.5 * v_rel, 0.0, 0.0]) + v_c9_vec - jnp.array([-v_c9,0,0])

    # impact parameter sampled uniformly up to (sigma_H2 + sigma_C9)/2 * 3  (arbitrary upper)
    b_max = 3.0 * sigma_mix
    b = jax.random.uniform(k_b, (), minval=0.0, maxval=b_max)

    # initial positions separated along x by 5*sigma_mix
    X1 = jnp.array([-2.5 * sigma_mix, 0.0, -b/2.0])
    X2 = jnp.array([+2.5 * sigma_mix, 0.0,  b/2.0])

    # H2 internal geometry
    X11_0 = jnp.array([0.0, 0.0, 0.5 * d_H2])
    X12_0 = jnp.array([0.0, 0.0, -0.5 * d_H2])

    # random orientation for H2 (rotation matrix) - hydrocarbon is spherical (no R)
    R1 = getRandRotMat(kR1, kR1)
    Xv11 = R1 @ X11_0
    Xv12 = R1 @ X12_0
    X11 = X1 + Xv11
    X12 = X1 + Xv12

    # hydrocarbon coordinates
    Xc = X2

    # angular velocities for H2 sampled from rotational thermal energy (equipartition)
    # rotational DOF for linear molecule: 2 DOF; sample angular components from Gaussian
    w1 = jax.random.normal(k_w1, (3,)) * jnp.sqrt(kB * T / I_h2)
    
    # hydrocarbon treated as spherical (I large) -> angular velocity is zero 
    w2 = jnp.zeros(3)

    m1 = m_H2
    m2 = m_C9H20

    # state for loop: positions, velocities, rotations, angular velocities, atom positions, step, dr
    state = (X1, X2, V1, V2, R1, w1, X11, X12, Xc, 0.0, 0)

    def cond_fn(state):
        X1, X2, V1, V2, R1, w1, X11, X12, Xc, dr, step = state
        return (dr <= 5.0 * sigma_mix) & (step < nSteps)

    def body_fn(state):
        X1, X2, V1, V2, R1, w1, X11, X12, Xc, dr, step = state
        step = step + 1
        dr = jnp.linalg.norm(X1 - X2)

        # compute pairwise forces: H2 atoms <-> C9 atom
        F13 = getFij(X11, Xc)
        F23 = getFij(X12, Xc)
        F1_total = F13 + F23
        F2_total = - (F13 + F23)   # hydrocarbon feels equal and opposite

        # Moments on H2 from forces (assume lever arms relative to H2 COM)
        # approximate simple torque around centre (z comp only matters for planar)
        r13 = X11 - X1
        r23 = X12 - X1
        M1 = jnp.cross(r13, F13) + jnp.cross(r23, F23)
        M2 = jnp.array([0.0, 0.0, 0.0])  # spherical hydrocarbon

        # Velocity Verlet translational
        V1_half = V1 + 0.5 * dt * getVdot(F1_total, m1)
        V2_half = V2 + 0.5 * dt * getVdot(F2_total, m2)
        X1_new = X1 + dt * V1_half
        X2_new = X2 + dt * V2_half

        # rotational update (H2)
        R1_half = R1 + 0.5 * dt * getRdot(w1, R1)
        w1_half = w1 + 0.5 * dt * getWdot(M1, I_h2)

        # full-step
        R1_new = R1 + dt * getRdot(w1_half, R1_half)
        # update H2 atom positions
        Xv11_new = R1_new @ X11_0
        Xv12_new = R1_new @ X12_0
        X11_new = X1_new + Xv11_new
        X12_new = X1_new + Xv12_new
        Xc_new = X2_new

        # recompute forces at t+dt
        F13_new = getFij(X11_new, Xc_new)
        F23_new = getFij(X12_new, Xc_new)
        F1_new = F13_new + F23_new
        F2_new = - (F13_new + F23_new)
        r13_new = X11_new - X1_new
        r23_new = X12_new - X1_new
        M1_new = jnp.cross(r13_new, F13_new) + jnp.cross(r23_new, F23_new)

        V1_new = V1_half + 0.5 * dt * getVdot(F1_new, m1)
        V2_new = V2_half + 0.5 * dt * getVdot(F2_new, m2)
        w1_new = w1_half + 0.5 * dt * getWdot(M1_new, I_h2)

        return (X1_new, X2_new, V1_new, V2_new, R1_new, w1_new,
                X11_new, X12_new, Xc_new, dr, step)

    X1f, X2f, V1f, V2f, R1f, w1f, X11f, X12f, Xcf, drf, stepf = jax.lax.while_loop(cond_fn, body_fn, state)

    # Energies initial / final
    Etr_init = 0.5 * m1 * jnp.linalg.norm(jnp.array([v_h2,0,0]))**2 + 0.5 * m2 * jnp.linalg.norm(jnp.array([-v_c9,0,0]))**2
    Erot1_init = 0.5 * I_h2 * (w1[0]**2 + w1[1]**2 + w1[2]**2)
    Etr_final = 0.5 * m1 * jnp.linalg.norm(V1f)**2 + 0.5 * m2 * jnp.linalg.norm(V2f)**2
    Erot1_final = 0.5 * I_h2 * (w1f[0]**2 + w1f[1]**2 + w1f[2]**2)

    # Return impact parameter (normalized by sigma_mix) and energies in kB units (K)
    return jnp.array([b / sigma_mix]), jnp.array([Etr_init / kB]), jnp.array([Erot1_init / kB]), jnp.array([Etr_final / kB]), jnp.array([Erot1_final / kB])

keys_all = jax.random.split(key, ncoll * 7).reshape(ncoll, 7, 2)

if ncoll == 1:
    results = simulate_one_collision(keys_all[0])
else:
    results = vmap(simulate_one_collision)(keys_all)

b_list, Etr_init_list, Er1_init_list, Etr_final_list, Er1_final_list = results

# Convert to numpy arrays and save
b_list = np.array(b_list).flatten()
Etr_init_list = np.array(Etr_init_list).flatten()
Er1_init_list = np.array(Er1_init_list).flatten()
Etr_final_list = np.array(Etr_final_list).flatten()
Er1_final_list = np.array(Er1_final_list).flatten()

df = pd.DataFrame({
    'b_over_sigma': b_list,
    'Etr_init_K': Etr_init_list,
    'Er1_init_K': Er1_init_list,
    'Etr_final_K': Etr_final_list,
    'Er1_final_K': Er1_final_list,
})

outname = f'collision_dataset_h2_c9h20_{ncoll}.csv'
df.to_csv(outname, index=False)
print(f"Saved dataset to {outname}")
