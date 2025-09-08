import numpy as np
import matplotlib.pyplot as plt

def LJ_energy(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

epsilon = [95.93, 34, 118, 119.5]  # N2, H2, O2, Ar in units of K!
# source: https://nvlpubs.nist.gov/nistpubs/jres/58/jresv58n2p93_A1b.pdf
sigma = [3.69, 3.06, 3.46, 3.42] # in units of Angstroms!

r = np.linspace(0.5, 10, 256)
for eps, sig, label in zip(epsilon, sigma, ['N$_2$', 'H$_2$', 'O$_2$', 'Ar']):
    plt.plot(r, LJ_energy(r, eps, sig), label=f'{label}: ε={eps}K, σ={sig}Å')

plt.xlabel(r'$r$/Å')
plt.ylabel(r'interaction energy / $k_b$ (K)')
plt.ylim(-120, 10)
plt.xlim(2, 8)
plt.grid(True)
plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.show()