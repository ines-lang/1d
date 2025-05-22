# 1D Synthetic PDE Dataset Generator

## Description

This project enables the generation of synthetic datasets for one-dimensional partial differential equations (PDEs) using customizable parameters. Currently, the only implemented PDE is the **Kuramoto–Sivashinsky (KS)** equation. The dataset is useful for studying spatiotemporal chaos and for training machine learning models for tasks such as forecasting or surrogate modeling of nonlinear dynamical systems.

GitHub repository: [https://github.com/ines-lang/1d](https://github.com/ines-lang/1d)

## Dataset Generation

The main script, `main.py`, allows the user to configure the simulation via parameters defined at the beginning of the file (USER INPUTS). The following PDE parameters can be adjusted:

- **PDE** to solve (currently only `ks`)
- **Initial condition** (`ic`)
- **Boundary condition** (`bc`)
- **Spatial domain** (`x_bounds`, `x_res`)
- **Time discretization** (`dt`, `t_end`, `save_frequency`)
- **Viscosity coefficient** (`nu`)
- **Number of simulations** (`simulations`)

## File Structure

The simulation pipeline is structured as follows:

- **`main.py`** – Entry point; handles user input and coordinates execution.
- **`generator.py`** – Parses parameters and initializes simulation.
- **`ks_jax_solver.py`** – Contains the PDE solver using the **Exponential Time Differencing Runge-Kutta 2nd order (ETDRK2)** method to solve the KS equation and compute \( u(x, t) \).

Upon execution, the code creates a directory structure based on the selected PDE and initial condition. For example:

1d/
└── ks/
└── random_ic_fourier_jax/
├── dataset.h5
└── plots/


Each simulation trajectory is stored in the HDF5 file (`dataset.h5`) with keys formatted as:
velocity_000, velocity_001, ..., velocity_NNN


Each dataset contains a 2D array with shape `(time, space)` and `dtype=float32`, representing the evolution of \( u(x,t) \) over time.

Additionally, the script randomly selects 10 simulations to visualize (this number can be modified at **line 111**) and plots their physical evolution of \( u(x,t) \) as a heatmap over time and space.

## Output Format

- **HDF5 datasets:** Named `velocity_{seed:03d}`.
- **Shape of each dataset:** `(time_steps, spatial_points)`
- **Data type:** `float32`

## Notes and Recommendations

- Among the available initial conditions, **`random_ic_fourier`** has been observed to provide the most stable and representative results for capturing the chaotic behavior of the Kuramoto–Sivashinsky equation.

- The solver is implemented in **JAX** and makes use of the **Fast Fourier Transform (FFT)** internally. Therefore, boundary conditions are not explicitly enforced during simulation. For consistency, it is recommended to set the boundary condition parameter to `periodic_bc`, which aligns with the use of FFT-based solvers.

## License

This project is released under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). You are free to use, modify, and distribute it with appropriate attribution.

---