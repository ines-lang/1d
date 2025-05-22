# PDE Dataset Generator
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import List

# =========================
# Boundary Conditions
# =========================

def periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

def dirichlet_bc(u, value=0.0):
    u[0] = u[-1] = value
    return u

def neumann_bc(u, q_left=0.0, q_right=0.0, dx=1.0):
    u[0] = u[1] - q_left * dx
    u[-1] = u[-2] + q_right * dx
    return u

# =========================
# Initial Conditions
# =========================

def sine_ic_np(domain_length, k=1.0):
    return -np.sin(k * np.pi * domain_length)

def sine_ic_jax(domain_length, k=1.0, amplitude=0.1):
    return -amplitude * jnp.sin(k * jnp.pi * domain_length)

def gaussian_ic_np(domain_length, mu=1.0, sigma=0.2, amplitude=1.0):
    return amplitude * np.exp(-(domain_length - mu) ** 2 / (2 * sigma ** 2))

def square_ic_np(domain_length, left=0.4, right=0.6, high=1.0, low=0.0):
    return np.where((domain_length >= left) & (domain_length <= right), high, low)

def sawtooth_ic_np(domain_length):
    return 2 * (domain_length - np.floor(domain_length + 0.5))

def random_ic_np(domain_length, seed=42, amplitude=1.0):
    np.random.seed(seed)
    return amplitude * (np.random.rand(len(domain_length)) - 0.5)

def random_ic_jax(domain_length, amplitude=1.0):
    key = jr.PRNGKey(42)  # Always fixed inside
    noise = jr.uniform(key, shape=domain_length.shape) - 0.5
    return amplitude * noise

def double_gaussian_ic_np(domain_length):
    return (
        gaussian_ic_np(domain_length, mu=0.5, sigma=0.05, amplitude=1.0)
        + gaussian_ic_np(domain_length, mu=1.5, sigma=0.05, amplitude=-1.0)
    )

def random_ic_fourier(domain_length, num_modes=10, seed=42, amplitude=1.0):
    np.random.seed(seed)
    u0 = np.zeros_like(domain_length)
    L = domain_length[-1] - domain_length[0]
    for _ in range(num_modes):
        k = np.random.randint(1, 6)  # low frequencies only
        phase = 2 * np.pi * np.random.rand()
        coeff = np.random.randn() / k  # decay spectrum
        u0 += coeff * np.sin(2 * np.pi * k * domain_length / L + phase)
    u0 = amplitude * u0
    u0 = np.clip(u0, -1.0, 1.0)  # avoid spikes
    return u0

def sine_ic_jax(domain_length, k=1.0, amplitude=0.1):
    return -amplitude * jnp.sin(k * jnp.pi * domain_length)

def gaussian_ic_jax(domain_length, mu=1.0, sigma=0.2, amplitude=1.0):
    return amplitude * jnp.exp(-((domain_length - mu) ** 2) / (2 * sigma ** 2))

def random_ic_fourier_jax(domain_length, seed, num_modes=20, amplitude=1.0):
    key = jr.PRNGKey(seed)
    u0 = jnp.zeros_like(domain_length)
    # Generate random wavenumbers, phases, and coefficients
    key_k, key_phase, key_coeff = jr.split(key, 3)
    k_values = jr.randint(key_k, shape=(num_modes,), minval=1, maxval=20)
    phases = 2 * jnp.pi * jr.uniform(key_phase, shape=(num_modes,))
    coeffs = jr.normal(key_coeff, shape=(num_modes,))
    L = domain_length[-1] - domain_length[0]
    for i in range(num_modes):
        k = k_values[i]
        phase = phases[i]
        coeff = coeffs[i]
        u0 += coeff * jnp.sin(2 * jnp.pi * k * domain_length / L + phase)
    u0 = amplitude * u0 / jnp.max(jnp.abs(u0))  # Normalize
    return u0

# =========================
# PDE Solvers (1D)
# =========================

def solve_burgers_1d(ic, bc, x_bounds, x_res, dt, t_end, save_freq, nu):
    xmin, xmax = x_bounds  # unpack the tuple
    domain_length = np.linspace(xmin, xmax, x_res)
    dx = domain_length[1] - domain_length[0]
    nt = int(t_end / dt)
    u = ic(domain_length)
    u_new = np.zeros_like(u)
    data = []
    for t in range(nt):
        u = bc(u)
        u_new[1:-1] = (
            u[1:-1]
            - dt / (2 * dx) * u[1:-1] * (u[2:] - u[:-2])
            + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        )
        u[:] = u_new[:]
        if t % save_freq == 0:
            data.append(u.copy())
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            print(f"Simulation blew up at t = {t * dt}")
            break

    return np.array(data), domain_length

# =========================
# Dataset Generator
# =========================

def generate_dataset(pde: str,
                      ic: callable, 
                      bc: callable, 
                      x_bounds: tuple,
                      x_res: int, 
                      dt: float, 
                      t_end: float, 
                      save_freq: int, 
                      nu: float, 
                      seed_list: List):
    
    if pde == "burgers_1d":
        return solve_burgers_1d(ic, bc, x_bounds, x_res, dt, t_end, save_freq, nu)
    elif pde == "ks":
        from ks_jax_solver import solve_ks_1d_jax
        return solve_ks_1d_jax(ic, bc, x_bounds, x_res, dt, t_end, save_freq, seed_list)
    else:
        raise ValueError(f"PDE '{pde}' not implemented.")