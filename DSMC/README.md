# DSMC-MDN

Direct Simulation Monte Carlo (DSMC) method with a surrogate neural network for the collision term. 

## Usage
### Arguments

You can customize the DSMC simulation by providing command-line arguments:

- `--mdn_model <path>`: Path to the trained MDN model.
- `--n_particles <int>`: Number of particles (default: 5000).
- `--n_steps <int>`: Number of steps (default: 1000).

Example usage:

Run the DSMC simulation using Python:

```powershell
python dsmc.py
```

Simulation results and figures will be saved in the `logs/` directory.


