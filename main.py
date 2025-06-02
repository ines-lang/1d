import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import h5py
import os
import random
import time

from generator import *
from ks_jax_solver import *

# =========================================
# USER INPUTS (PDE parameters)
# =========================================
"""
Simulation parameters:

pde : str
    PDE to solve. Options: 'burgers_1d', 'ks', 'kdv_1d_jax', 'kdv_1d_jax_vhat'

ic : callable
    Initial condition function. Options: random_ic_fourier_jax, ??

bc : callable
    Boundary condition. Options: periodic_bc, dirichlet_bc.
    For PDEs using jax, the boundary condition is not used. 
    As JAX computes spatial derivatives using the Fast Fourier Transform (FFT), which assumes periodicity

x_bounds : tuple
    Spatial domain as (xmin, xmax)

x_res : int
    Number of spatial points

dt : float
    Time step size

t_end : float
    Final simulation time

save_freq : int
    Save data every this many steps

nu : float
    Viscosity (used only for Burgers)

simulations : int
    Number of simulations to run
"""

pde = "burgers"
ic = random_ic_fourier_jax
bc = periodic_bc 

x_bounds = (0.0, 128.0)
x_res = 128 
dt = 0.001 
t_end = 100.0 
save_freq = 10
nu = 0.5
simulations = 50

# =========================================
# GENERATE AND SAVE DATASET
# =========================================
seed_list = list(range(simulations)) 
dataset_all_trajs, domain_length = generate_dataset(
    pde=pde,
    ic=ic,
    bc=bc,
    x_bounds=x_bounds,
    x_res=x_res,
    dt=dt,
    t_end=t_end,
    save_freq=save_freq,
    nu=nu,
    seed_list=seed_list)

print("Original shape of the data from the solver:", dataset_all_trajs.shape)

#  Directory dependant on the pde and initial condition
base_dir = os.path.join("1d", pde, ic.__name__)
os.makedirs(base_dir, exist_ok=True)
file_name = "dataset.h5"
data_path = os.path.join(base_dir, file_name)
os.makedirs(os.path.dirname(data_path), exist_ok=True)

# Create the h5py file and save the dataset
with h5py.File(data_path, "w") as h5file:
    for sim_idx in range(len(seed_list)):
        seed = seed_list[sim_idx]
        u_xt = dataset_all_trajs[sim_idx, :, :, 0]  # 0 is valid as we have only one channel
        dataset_name = f'velocity_{seed:03d}'
        h5file.create_dataset(dataset_name, data=u_xt)  
    print(f"File created at {data_path}")
    # Print structure 
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
    h5file.visititems(print_structure)

# Normalize the dataset

#Full space-time evolution plot
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

random.seed(time.time())
selected_indices = random.sample(range(len(seed_list)), 10) # Plotting just X random seeds
for sim_idx in selected_indices:
    seed = seed_list[sim_idx]
    plt.figure(figsize=(10, 4))
    t_array = np.linspace(0, t_end, dataset_all_trajs.shape[1]) # changed from -1 to 1
    x_array = domain_length
    u_xt = dataset_all_trajs[sim_idx, :, :, 0]  # 0 is valid as we have only one channel
    vabs = np.max(np.abs(u_xt))
    plt.imshow(u_xt.T, cmap='RdBu', origin='lower',
               extent=(t_array[0], t_array[-1], x_array[0], x_array[-1]),
               aspect='auto',
               vmin=-vabs, vmax=vabs)
    plt.colorbar(label='u(x, t)')
    plt.xlabel("Time step")
    plt.ylabel("Space")
    plt.title(f"{ic.__name__} - {bc.__name__}- seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"seed_{seed}.png"), dpi=150)
    plt.close()