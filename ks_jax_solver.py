import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import h5py

class KS_ETDRK2():
    def __init__(self, L, N, dt):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N

        wavenumbers = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
        self.derivative_operator = 1j * wavenumbers

        linear_operator = - self.derivative_operator**2 - self.derivative_operator**4
        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef_1 = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )
        self.coef_2 = jnp.where(
            linear_operator == 0.0,
            dt / 2,
            (self.exp_term - 1.0 - linear_operator * dt) / (linear_operator**2 * dt)
        )

        self.alias_mask = (wavenumbers < 2/3 * jnp.max(wavenumbers))
    
    def __call__(self, u):
        u_nonlin = - 0.5 * u**2
        u_hat = jnp.fft.rfft(u)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_stage_1_hat = self.exp_term * u_hat + self.coef_1 * u_nonlin_der_hat
        u_stage_1 = jnp.fft.irfft(u_stage_1_hat, n=self.N)

        u_stage_1_nonlin = - 0.5 * u_stage_1**2
        u_stage_1_nonlin_hat = jnp.fft.rfft(u_stage_1_nonlin)
        u_stage_1_nonlin_hat = self.alias_mask * u_stage_1_nonlin_hat
        u_stage_1_nonlin_der_hat = self.derivative_operator * u_stage_1_nonlin_hat

        u_next_hat = u_stage_1_hat + self.coef_2 * (u_stage_1_nonlin_der_hat - u_nonlin_der_hat)
        u_next = jnp.fft.irfft(u_next_hat, n=self.N)

        return u_next

def solve_ks_1d_jax(ic, bc, x_bounds, x_res, dt, t_end, save_freq, seed_list):

    data_all_trajs = []
    L = x_bounds[1] - x_bounds[0]
    domain_length = jnp.linspace(x_bounds[0], x_bounds[1], x_res, endpoint=False)

    for seed in seed_list:
        
        u0 = ic(domain_length,seed)

        model = KS_ETDRK2(L, x_res, dt)
        step = jit(model)

        u = u0
        data = [u]
        nt = int(t_end / dt)

        for t in range(nt):
            u = step(u)
            #check for nan values
            if jnp.any(jnp.isnan(u)) or jnp.any(jnp.isinf(u)):
                print(f"Simulation blew up at t = {t * dt}")
                break
            if t % save_freq == 0:
                data.append(u)
        #append the channel dimension to the dataset
        data=np.array(data) 
        data = np.expand_dims(data, axis=-1) # new final dimension for the channel
        data_all_trajs.append(data) # shape: (n_sim, t, x, 1)
    data_all_trajs = np.array(data_all_trajs)  # shape: (n_sim, t, x, 1)
    assert data_all_trajs.ndim == 4, "Output shape must be (n_sim, t, x, 1)"
    assert data_all_trajs.shape[-1] == 1, "Last dim must be channel=1"
    if data.shape[1] == 1 and data.shape[2] != 1:
        # likely data came as (n_sim, t, 1, x) → fix it
        data = np.transpose(data, (0, 2, 1))  # → (n_sim, t, x, 1)

    return data_all_trajs, np.array(domain_length)