#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ferroelectric Phase-Field Simulation using FFT (Semi-Implicit Fourier Spectral Method)
=========================================================================================

Methodology:
    - Order parameter: vector polarization P = (Px, Py) in 2D
    - TDGL:    dP/dt = -Gamma * dF/dP
    - Poisson: nabla^2 Phi = div(P)         -> electrostatic (depolarization) field
    - Free energy density:
        f = f_bulk (Landau-Devonshire)
          + f_grad (Lifshitz invariants)
          + f_elec (depolarization)
          [+ f_elastic + f_electrostr  -- omitted here for clarity, see notes below]

Discretization:
    - FFT (semi-implicit Fourier spectral) on a uniform periodic grid
    - The Lifshitz gradient + depolarization terms combine into a 2x2 matrix
      operator M(k) acting on the Fourier transform of (Px, Py).
    - Poisson is solved "for free" by one division by -k^2.

The scalar Laplacian  -K2*u_hat  generalizes to a matrix operator  M(k) . P_hat .
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import ScalarFormatter

# ------------------- GPU SETUP (CuPy / NumPy) -------------------
GPU_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    cp.zeros(1)
    GPU_AVAILABLE = True
except Exception:
    import numpy as cp
    GPU_AVAILABLE = False

# ------------------- PAGE -------------------
st.set_page_config(page_title="Ferroelectric FFT Phase-Field", layout="wide")
st.title("Ferroelectric Phase-Field — FFT (Semi-Implicit Fourier Spectral)")

st.markdown("""
**Methodology**: TDLGD equation + Landau–Devonshire bulk + Lifshitz gradient
+ Poisson electrostatics. **Discretization**: FFT on a periodic grid.
""")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})

# ------------------- SIDEBAR -------------------
st.sidebar.header("Grid")
max_res = 1024 if GPU_AVAILABLE else 512
Nx = st.sidebar.slider("Nx", 64, max_res, 256, 32)
Ny = st.sidebar.slider("Ny", 64, max_res, 256, 32)
L_box = st.sidebar.number_input("Box size L (nd)", 0.5, 10.0, 1.0, 0.1)

st.sidebar.header("Time stepping")
dt = st.sidebar.number_input("dt (nd)", 1e-4, 1e-1, 1e-2, format="%.4f")
n_steps = st.sidebar.slider("Steps", 50, 20000, 2000, 50)
save_every = st.sidebar.slider("Save every", 1, 200, max(1, 2000 // 30), 1)

st.sidebar.header("Landau–Devonshire (bulk)")
alpha1  = st.sidebar.number_input("α₁  (<0 below Tc)", -5.0, 0.0, -1.0, 0.1)
alpha11 = st.sidebar.number_input("α₁₁ (>0)", 0.01, 10.0, 1.0, 0.05)
alpha12 = st.sidebar.number_input("α₁₂", -5.0, 5.0, 1.0, 0.1)
alpha111= st.sidebar.number_input("α₁₁₁ (6th-order, optional)", 0.0, 5.0, 0.0, 0.1)

st.sidebar.header("Lifshitz gradient coefficients")
G11 = st.sidebar.slider("G₁₁", 0.01, 5.0, 1.0, 0.01)
G12 = st.sidebar.slider("G₁₂", -2.0, 2.0, 0.0, 0.01)
G44 = st.sidebar.slider("G₄₄", 0.01, 5.0, 0.5, 0.01)

st.sidebar.header("Kinetics")
Gamma = st.sidebar.slider("Γ (mobility)", 0.01, 5.0, 1.0, 0.01)

st.sidebar.header("Electrostatics")
use_elec = st.sidebar.checkbox("Enable depolarization field (Poisson)", True)
use_ext_E = st.sidebar.checkbox("Apply external E field", False)
if use_ext_E:
    Ex_ext = st.sidebar.slider("E_ext,x", -2.0, 2.0, 0.5, 0.05)
    Ey_ext = st.sidebar.slider("E_ext,y", -2.0, 2.0, 0.0, 0.05)
else:
    Ex_ext = Ey_ext = 0.0

st.sidebar.header("Initial condition")
noise_amp = st.sidebar.slider("Noise amplitude", 1e-4, 1e-1, 1e-2, format="%.4f")
seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

run_button = st.sidebar.button("Run simulation")

# ------------------- HELPERS -------------------
def to_cpu(arr):
    return cp.asnumpy(arr) if GPU_AVAILABLE else arr

def to_gpu(arr):
    return cp.asarray(arr) if GPU_AVAILABLE else arr

# ------------------- K-SPACE OPERATORS -------------------
def build_k_operators(Nx, Ny, L):
    """Build the matrix operator M(k) = M_grad + M_elec acting on (Px_hat, Py_hat).

    Lifshitz gradient (2D):
        f_grad = (G11/2)(Px,x^2 + Py,y^2) + G12 Px,x Py,y + (G44/2)(Px,y + Py,x)^2

    Variational derivative in Fourier space gives the 2x2 matrix M_grad.
    Adding the electrostatic projector k⊗k/k^2 gives M(k).
    """
    dx = L / Nx
    dy = L / Ny
    kx = cp.fft.fftfreq(Nx, d=dx) * 2 * cp.pi
    ky = cp.fft.fftfreq(Ny, d=dy) * 2 * cp.pi
    KX, KY = cp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2_safe = cp.where(K2 == 0, 1.0, K2)  # avoid div-by-zero; will zero out DC below

    # Lifshitz gradient operator
    M_grad_xx = G11 * KX**2 + G44 * KY**2
    M_grad_yy = G44 * KX**2 + G11 * KY**2
    M_grad_xy = (G12 + G44) * KX * KY

    # Electrostatic (depolarization) projector  k⊗k/k^2
    if use_elec:
        M_elec_xx = KX**2 / K2_safe
        M_elec_yy = KY**2 / K2_safe
        M_elec_xy = KX * KY / K2_safe
    else:
        M_elec_xx = M_elec_yy = M_elec_xy = 0.0

    M_xx = M_grad_xx + M_elec_xx
    M_yy = M_grad_yy + M_elec_yy
    M_xy = M_grad_xy + M_elec_xy

    # Zero mode: no gradient, no depolarization (uniform P has no spatial cost)
    M_xx[0, 0] = 0.0
    M_yy[0, 0] = 0.0
    M_xy[0, 0] = 0.0

    return KX, KY, K2_safe, M_xx, M_yy, M_xy, dx, dy

# ------------------- POISSON SOLVER -------------------
def solve_poisson(Px, Py, KX, KY, K2_safe):
    """Solve  nabla^2 Phi = div(P),  return E = -grad(Phi)."""
    Px_hat = cp.fft.fft2(Px)
    Py_hat = cp.fft.fft2(Py)
    div_P_hat = 1j * (KX * Px_hat + KY * Py_hat)  # i k . P_hat
    Phi_hat = -div_P_hat / K2_safe                 # nabla^2 -> -k^2
    Phi_hat[0, 0] = 0.0                            # zero-mean potential
    Ex_hat = -1j * KX * Phi_hat
    Ey_hat = -1j * KY * Phi_hat
    Ex = cp.fft.ifft2(Ex_hat).real
    Ey = cp.fft.ifft2(Ey_hat).real
    return Ex, Ey

# ------------------- LANDAU-DERIVONSHIRE BULK -------------------
def landau_deriv(Px, Py):
    """d f_bulk / d P  for tetragonal Landau-Devonshire:
        f_bulk = a1(P^2) + a11(Px^4+Py^4) + a12 Px^2 Py^2 + a111(Px^6+Py^6) + ...
    Returns (dF/dPx, dF/dPy) — the NONLINEAR part, computed in real space.
    """
    dPx = (2*alpha1*Px
           + 4*alpha11*Px**3
           + 2*alpha12*Px*Py**2
           + 6*alpha111*Px**5)
    dPy = (2*alpha1*Py
           + 4*alpha11*Py**3
           + 2*alpha12*Py*Px**2
           + 6*alpha111*Py**5)
    return dPx, dPy

# ------------------- SIMULATION CORE -------------------
def run_simulation():
    cp.random.seed(seed)
    KX, KY, K2_safe, M_xx, M_yy, M_xy, dx, dy = build_k_operators(Nx, Ny, L_box)

    # Initial condition: small random noise around P = 0 (paraelectric state)
    Px = noise_amp * (cp.random.rand(Nx, Ny) - 0.5)
    Py = noise_amp * (cp.random.rand(Nx, Ny) - 0.5)

    snapshots, energy_log = [], []
    Ps_eq = cp.sqrt(-alpha1 / (2*alpha11)) if alpha1 < 0 and alpha11 > 0 else 1.0

    for step in range(n_steps + 1):
        # --- Nonlinear part of dF/dP (Landau bulk), computed in real space ---
        dF_bulk_x, dF_bulk_y = landau_deriv(Px, Py)

        # --- Electrostatic field from Poisson (linear but treated as part of NL driving) ---
        if use_elec:
            Ex_dep, Ey_dep = solve_poisson(Px, Py, KX, KY, K2_safe)
            # dF_elec/dP = -E_dep  (depolarization field opposes P divergence)
            dF_elec_x = -Ex_dep
            dF_elec_y = -Ey_dep
        else:
            dF_elec_x = dF_elec_y = 0.0

        # External field contribution to dF/dP is -E_ext (constant in space)
        NL_x = dF_bulk_x + dF_elec_x - Ex_ext
        NL_y = dF_bulk_y + dF_elec_y - Ey_ext

        # --- Semi-implicit Fourier spectral step ---
        #   [I + dt*Gamma*M] P^{n+1}_hat = P^n_hat - dt*Gamma*NL_hat
        Px_hat = cp.fft.fft2(Px)
        Py_hat = cp.fft.fft2(Py)
        NLx_hat = cp.fft.fft2(NL_x)
        NLy_hat = cp.fft.fft2(NL_y)

        rhs_x = Px_hat - dt * Gamma * NLx_hat
        rhs_y = Py_hat - dt * Gamma * NLy_hat

        A = 1.0 + dt * Gamma * M_xx
        D = 1.0 + dt * Gamma * M_yy
        B = dt * Gamma * M_xy
        det = A * D - B * B

        Px_hat_new = (D * rhs_x - B * rhs_y) / det
        Py_hat_new = (-B * rhs_x + A * rhs_y) / det

        Px = cp.fft.ifft2(Px_hat_new).real
        Py = cp.fft.ifft2(Py_hat_new).real

        # --- Diagnostics: energy components ---
        if step % save_every == 0 or step == n_steps:
            f_bulk = (alpha1*(Px**2 + Py**2)
                      + alpha11*(Px**4 + Py**4)
                      + alpha12*Px**2*Py**2
                      + alpha111*(Px**6 + Py**6))
            Px_x, Px_y = cp.gradient(Px, dx, dy)
            Py_x, Py_y = cp.gradient(Py, dx, dy)
            f_grad = (0.5*G11*(Px_x**2 + Py_y**2)
                      + G12*Px_x*Py_y
                      + 0.5*G44*(Px_y + Py_x)**2)
            if use_elec:
                Ex_dep, Ey_dep = solve_poisson(Px, Py, KX, KY, K2_safe)
                f_elec = -0.5 * (Ex_dep*Px + Ey_dep*Py)
            else:
                f_elec = 0.0
            total = float(cp.mean(f_bulk + f_grad + f_elec))
            energy_log.append((step, float(cp.mean(f_bulk)),
                               float(cp.mean(f_grad)), float(cp.mean(f_elec) if use_elec else 0.0),
                               total))
            snapshots.append((step, to_cpu(Px), to_cpu(Py)))

    return snapshots, energy_log, Ps_eq

# ------------------- DOMAIN COLOR VISUALIZATION -------------------
def polarization_color(Px, Py, Ps):
    """HSV coloring: hue = direction of P, value = |P|/Ps."""
    angle = np.arctan2(Py, Px)         # -pi..pi
    hue = (angle + np.pi) / (2*np.pi)  # 0..1
    mag = np.sqrt(Px**2 + Py**2)
    sat = np.clip(mag / (Ps + 1e-12), 0, 1)
    hsv = np.stack([hue, np.ones_like(hue)*0.9, sat], axis=-1)
    return hsv_to_rgb(hsv)

# ------------------- RUN + VISUALIZE -------------------
if run_button:
    with st.spinner(f"Running FFT phase-field on {'GPU' if GPU_AVAILABLE else 'CPU'}..."):
        snaps, energy_log, Ps_eq = run_simulation()
    st.success(f"Done. {len(snaps)} frames saved. Equilibrium |Ps| ≈ {float(to_cpu(Ps_eq)):.3f}")

    # ----- Final snapshot -----
    st.header("Final polarization field")
    step_f, Px_f, Py_f = snaps[-1]
    rgb = polarization_color(Px_f, Py_f, float(to_cpu(Ps_eq)))

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(rgb, origin='lower', extent=[0, L_box, 0, L_box])
        ax.set_title(f"P @ step {step_f}   (color = direction, brightness = |P|)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        # Color wheel
        ax_in = fig.add_axes([0.78, 0.78, 0.12, 0.12], polar=True)
        theta = np.linspace(0, 2*np.pi, 100)
        ax_in.scatter(theta, np.ones_like(theta),
                      c=hsv_to_rgb(np.stack([(theta/(2*np.pi)), np.ones_like(theta)*0.9,
                                             np.ones_like(theta)]).T),
                      s=8)
        ax_in.set_yticks([]); ax_in.set_xticks([])
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(Px_f, origin='lower', cmap='RdBu_r',
                       vmin=-float(to_cpu(Ps_eq)), vmax=float(to_cpu(Ps_eq)))
        plt.colorbar(im, ax=ax, label="Pₓ")
        ax.set_title(f"Pₓ @ step {step_f}")
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(Py_f, origin='lower', cmap='RdBu_r',
                       vmin=-float(to_cpu(Ps_eq)), vmax=float(to_cpu(Ps_eq)))
        plt.colorbar(im, ax=ax, label="Pᵧ")
        ax.set_title(f"Pᵧ @ step {step_f}")
        st.pyplot(fig)

    # ----- Energy evolution -----
    st.header("Energy evolution")
    energy_log_arr = np.array(energy_log)
    steps = energy_log_arr[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, energy_log_arr[:, 1], label="f_bulk (Landau)")
    ax.plot(steps, energy_log_arr[:, 2], label="f_grad (Lifshitz)")
    if use_elec:
        ax.plot(steps, energy_log_arr[:, 3], label="f_elec (depolarization)")
    ax.plot(steps, energy_log_arr[:, 4], 'k-', lw=2.5, label="Total f")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy density (nd)")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ----- Frame scrubber -----
    st.header("Time evolution playback")
    frame = st.slider("Frame", 0, len(snaps) - 1, len(snaps) - 1)
    s_i, Px_i, Py_i = snaps[frame]
    rgb_i = polarization_color(Px_i, Py_i, float(to_cpu(Ps_eq)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_i, origin='lower', extent=[0, L_box, 0, L_box])
    ax.set_title(f"P @ step {s_i}")
    st.pyplot(fig)

    # ----- CSV export -----
    import pandas as pd
    df = pd.DataFrame(energy_log, columns=["step", "f_bulk", "f_grad", "f_elec", "f_total"])
    st.dataframe(df.head(20).style.format("{:.4e}"))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download energy CSV", csv, "ferro_energy.csv", "text/csv")

else:
    st.info("Configure parameters in the sidebar and click **Run simulation**.")

# ------------------- NOTES -------------------
st.markdown("""
---
**Notes & extensions**

* **Periodic BCs only** — FFT mandates periodicity. For arbitrary geometries
  (nanoparticles, curved surfaces) a finite-element discretization is more appropriate.
* **Elastic + electrostrictive coupling** — omitted above for clarity. In a full
  ferroelectric simulation you add:
    f_elastic = 0.5 * C_ijkl * (eps_ij - eps^0_ij(P))(eps_kl - eps^0_kl(P))
  where `eps^0_ij(P) = Q_ijkl * P_k * P_l` is the electrostrictive eigenstrain.
  This adds a *P²-derivative* term to NL(P), computed in real space. The
  mechanical equilibrium `div sigma = 0` can also be solved in Fourier for the
  homogeneous-elasticity case.
* **Why semi-implicit?** — Explicit Euler for `dP/dt = Gamma * G11 * nabla^2 P`
  has stability limit `dt < dx^2 / (4 * Gamma * G11)`. The semi-implicit scheme
  `(I + dt*Gamma*M) P^{n+1} = P^n + ...` is unconditionally stable for the
  linear part, allowing dt ~100–1000x larger.
""")
