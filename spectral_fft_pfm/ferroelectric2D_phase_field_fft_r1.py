#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ferroelectric Phase-Field Simulation using FFT (Semi-Implicit Fourier Spectral)
=================================================================================

3-component polarization P = (Px, Py, Pz) in a 2D in-plane grid.
Thin-film geometry: periodic in (x,y), open-circuit in z.

Methodology:
    - TDGL:    dP/dt = -Gamma * dF/dP
    - Poisson (in-plane, 2D): nabla^2 Phi = div(P_inplane)   -> E_inplane_dep
    - Surface depolarization for Pz via thin-film kernel:
          n(k) = tanh(k*h/2) / (k*h/2)
      (open-circuit: uniform Pz fully penalized, fine stripes unpenalized)
    - Free energy density:
        f = f_bulk (Landau-Devonshire, 3-component tetragonal)
          + f_grad (Lifshitz invariants, 2D-reduced)
          + f_depol_inplane (2D projector on Px, Py)
          + f_depol_Pz (thin-film kernel on Pz)

Discretization:
    - FFT (semi-implicit Fourier spectral) on uniform periodic grid
    - Block-diagonal 3x3 linear operator M(k):
        2x2 in-plane block (Px, Py): Lifshitz gradient + 2D depol projector
        1x1 Pz block:               G44*k_par^2 + n(k)/eps0 * screening_factor

All quantities in SI units: P [C/m^2], E [V/m], length [m], time [s].
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ------------------- PHYSICAL CONSTANTS (SI) -------------------
EPSILON_0 = 8.854187817e-12  # F/m

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
st.set_page_config(page_title="Ferroelectric FFT Phase-Field (Thin Film)", layout="wide")
st.title("Ferroelectric Phase-Field — FFT, 3-Component P, Thin-Film Depolarization")

st.markdown("""
**Methodology**: TDLGD + Landau-Devonshire bulk + Lifshitz gradient +
in-plane depolarization (2D projector) + **out-of-plane depolarization via thin-film kernel**.

All quantities in SI units. P_z is now an active order parameter with open-circuit
boundary conditions (top/bottom surfaces unscreened by default).
""")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})

# ------------------- SIDEBAR: GEOMETRY (SI) -------------------
st.sidebar.header("Geometry (SI units)")
max_res = 1024 if GPU_AVAILABLE else 512
Nx = st.sidebar.slider("Nx", 64, max_res, 256, 32)
Ny = st.sidebar.slider("Ny", 64, max_res, 256, 32)
L_box_nm = st.sidebar.number_input("Box size L (nm)", 10.0, 500.0, 64.0, 2.0)
L_box = L_box_nm * 1e-9  # m

st.sidebar.header("Thin-film thickness (open-circuit direction)")
h_film_nm = st.sidebar.number_input("Film thickness h (nm)", 1.0, 50.0, 8.0, 0.5)
h_film = h_film_nm * 1e-9  # m

# ------------------- TIME STEPPING (SI) -------------------
st.sidebar.header("Time stepping (SI)")
dt_ps = st.sidebar.number_input("dt (ps)", 0.001, 100.0, 1.0, 0.01)
dt = dt_ps * 1e-12  # s
n_steps = st.sidebar.slider("Steps", 50, 50000, 5000, 50)
save_every = st.sidebar.slider("Save every", 1, 500, max(1, 5000 // 30), 1)

# ------------------- LANDAU-DEVONSHIRE (SI) -------------------
st.sidebar.header("Landau-Devonshire (SI: V·m/C, V·m⁵/C³)")
preset = st.sidebar.selectbox("Material preset", ["PbTiO3-like", "HfO2-like", "Custom"], index=0)
if preset == "PbTiO3-like":
    alpha1_def, alpha11_def, alpha12_def = -3.0e10, 6.0e10, 1.5e11
    G11_def, G44_def = 1.0e-8, 1.0e-8
elif preset == "HfO2-like":
    alpha1_def, alpha11_def, alpha12_def = -2.5e9, 6.0e10, 1.5e11
    G11_def, G44_def = 1.0e-10, 1.0e-10
else:
    alpha1_def, alpha11_def, alpha12_def = -3.0e10, 6.0e10, 1.5e11
    G11_def, G44_def = 1.0e-8, 1.0e-8

alpha1  = st.sidebar.number_input("α₁ (V·m/C, <0 below Tc)", -5e11, 0.0, float(alpha1_def), 1e9, format="%.3e")
alpha11 = st.sidebar.number_input("α₁₁ (V·m⁵/C³)", 1e8, 1e12, float(alpha11_def), 1e9, format="%.3e")
st.sidebar.markdown("*For tetragonal symmetry: α₁₂ > 2α₁₁*")
alpha12 = st.sidebar.number_input("α₁₂ (V·m⁵/C³)", 0.0, 5e11, float(alpha12_def), 1e9, format="%.3e")

# ------------------- GRADIENT (SI) -------------------
st.sidebar.header("Lifshitz gradient (V·m³/C)")
G11 = st.sidebar.number_input("G₁₁", 1e-11, 1e-6, float(G11_def), 1e-10, format="%.3e")
G12 = 0.0
G44 = st.sidebar.number_input("G₄₄", 1e-11, 1e-6, float(G44_def), 1e-10, format="%.3e")

# ------------------- SUBSTRATE ANISOTROPY (mimics compressive strain) -------------------
st.sidebar.header("Substrate anisotropy (favors Pz)")
st.sidebar.markdown("""
Adds quadratic penalty β_sub*(Px²+Py²)/2 on in-plane P.
With β_sub > |α₁|, in-plane P is suppressed and only Pz can be ferroelectric.
This mimics compressive substrate strain (which favors out-of-plane Pz)
without requiring full elastic + electrostrictive coupling.
""")
use_substrate = st.sidebar.checkbox("Enable substrate anisotropy", True)
if use_substrate:
    beta_sub = st.sidebar.number_input(
        "β_sub (V·m/C, need > |α₁| = {:.2e})".format(abs(alpha1)),
        0.0, 1e12, float(abs(alpha1) * 1.5), 1e9, format="%.3e"
    )
else:
    beta_sub = 0.0

# ------------------- KINETICS -------------------
st.sidebar.header("Kinetics")
Gamma = st.sidebar.number_input("Γ (Ω·m)⁻¹", 0.001, 100.0, 1.0, 0.1)

# ------------------- ELECTROSTATICS -------------------
st.sidebar.header("Depolarization (open-circuit kernel)")
use_elec = st.sidebar.checkbox("Enable depolarization fields", True)
st.sidebar.markdown("**Screening factor** (0 = perfect electrodes / short-circuit, "
                    "1 = open-circuit / no electrodes):")
screening_factor = st.sidebar.slider("Screening factor s ∈ [0,1]", 0.0, 1.0, 0.65, 0.01)
st.sidebar.markdown("*With s≈0.6–0.7, uniform Pz is penalized but stripe Pz patterns "
                    "are favored → 180° stripe domains form.*")

use_ext_E = st.sidebar.checkbox("Apply external E field", False)
if use_ext_E:
    Ex_ext = st.sidebar.number_input("E_ext,x (MV/m)", -500.0, 500.0, 50.0, 1.0) * 1e6
    Ey_ext = st.sidebar.number_input("E_ext,y (MV/m)", -500.0, 500.0, 0.0, 1.0) * 1e6
    Ez_ext = st.sidebar.number_input("E_ext,z (MV/m)", -500.0, 500.0, 0.0, 1.0) * 1e6
else:
    Ex_ext = Ey_ext = Ez_ext = 0.0

# ------------------- INITIAL CONDITION -------------------
st.sidebar.header("Initial condition")
init_mode = st.sidebar.selectbox(
    "Init type",
    ["Band-limited noise on Pz", "Pure white noise (all 3 components)", "Seeded Pz stripes"],
    index=0
)
noise_amp_uc = st.sidebar.number_input("Noise amplitude (μC/cm²)", 1.0, 100.0, 20.0, 1.0)
noise_amp = noise_amp_uc * 1e4 * 1e-4  # μC/cm² -> C/m² (1 μC/cm² = 1e-2 C/m²)
noise_cutoff = st.sidebar.slider("Band cutoff (frac of Nyquist)", 0.05, 0.5, 0.15, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

# ------------------- VECTOR DISPLAY -------------------
st.sidebar.header("Vector display")
show_arrows = st.sidebar.checkbox("Overlay in-plane P arrows", True)
arrow_skip = st.sidebar.slider("Arrow subsample stride", 4, 40, 12, 2)

run_button = st.sidebar.button("Run simulation")

# ------------------- DERIVED QUANTITIES -------------------
Ps_eq = float(np.sqrt(-alpha1 / (2*alpha11))) if alpha1 < 0 and alpha11 > 0 else 1.0
xi_wall = float(np.sqrt(G11 / abs(alpha1))) if alpha1 != 0 else 1.0
st.sidebar.markdown(f"**Derived**: |Ps| = {Ps_eq*100:.2f} μC/cm², "
                    f"wall width ξ = {xi_wall*1e9:.3f} nm")

# ------------------- HELPERS -------------------
def to_cpu(arr):
    return cp.asnumpy(arr) if GPU_AVAILABLE else arr

def to_gpu(arr):
    return cp.asarray(arr) if GPU_AVAILABLE else arr

# ------------------- INITIAL CONDITION -------------------
def make_initial_P(Nx, Ny, L, mode, amp, cutoff, ps_eq):
    cp.random.seed(seed)
    kx_n = cp.fft.fftfreq(Nx)
    ky_n = cp.fft.fftfreq(Ny)
    KXn, KYn = cp.meshgrid(kx_n, ky_n, indexing='ij')
    kr = cp.sqrt(KXn**2 + KYn**2)

    if mode == "Pure white noise (all 3 components)":
        Px = amp * (cp.random.rand(Nx, Ny) - 0.5)
        Py = amp * (cp.random.rand(Nx, Ny) - 0.5)
        Pz = amp * (cp.random.rand(Nx, Ny) - 0.5)
        return Px, Py, Pz

    if mode == "Seeded Pz stripes":
        x = cp.linspace(0, L, Nx, endpoint=False)[:, None]
        stripes = cp.sign(cp.cos(2 * cp.pi * x * 4 / L))
        Pz = ps_eq * stripes * cp.ones((Nx, Ny))
        Px = 0.02 * ps_eq * (cp.random.rand(Nx, Ny) - 0.5)
        Py = 0.02 * ps_eq * (cp.random.rand(Nx, Ny) - 0.5)
        return Px, Py, Pz

    # Band-limited noise on Pz (default)
    mask = (kr <= cutoff).astype(cp.float64) * cp.exp(-kr**2 / (2 * (cutoff * 1.2)**2))
    Pz_raw = (cp.random.rand(Nx, Ny) - 0.5)
    Pz_bl = cp.fft.ifft2(mask * cp.fft.fft2(Pz_raw)).real
    Pz_bl /= (cp.std(Pz_bl) + 1e-12)
    Pz = amp * Pz_bl
    # Small noise on Px, Py
    Px_raw = (cp.random.rand(Nx, Ny) - 0.5)
    Py_raw = (cp.random.rand(Nx, Ny) - 0.5)
    Px_bl = cp.fft.ifft2(mask * cp.fft.fft2(Px_raw)).real
    Py_bl = cp.fft.ifft2(mask * cp.fft.fft2(Py_raw)).real
    Px_bl /= (cp.std(Px_bl) + 1e-12)
    Py_bl /= (cp.std(Py_bl) + 1e-12)
    Px = 0.05 * amp * Px_bl
    Py = 0.05 * amp * Py_bl
    return Px, Py, Pz

# ------------------- K-SPACE OPERATORS -------------------
def build_k_operators(Nx, Ny, L, h_film):
    """Build the 3x3 block-diagonal linear operator M(k) for (Px, Py, Pz).

    In-plane 2x2 block (Px, Py):
        M_grad_xx = G11*kx^2 + G44*ky^2 + (2D depol projector) kx^2/k^2
        M_grad_yy = G44*kx^2 + G11*ky^2 + (2D depol projector) ky^2/k^2
        M_grad_xy = (G12+G44)*kx*ky + (2D depol projector) kx*ky/k^2

    Pz 1x1 block:
        M_zz = G44 * k_par^2 + n(k_par)/eps0 * screening_factor
    where n(k) = tanh(k*h/2)/(k*h/2) is the thin-film depolarization kernel
    (open-circuit: n(0)=1 full depol for uniform Pz, n(inf)=0 for fine stripes).
    """
    dx = L / Nx
    dy = L / Ny
    kx = cp.fft.fftfreq(Nx, d=dx) * 2 * cp.pi
    ky = cp.fft.fftfreq(Ny, d=dy) * 2 * cp.pi
    KX, KY = cp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2_safe = cp.where(K2 == 0, 1.0, K2)
    K_par = cp.sqrt(K2)

    # Thin-film depolarization kernel for Pz (open-circuit)
    kh = K_par * h_film / 2.0
    # n(k) = tanh(kh)/kh, with n(0)=1
    n_kernel = cp.where(K_par == 0, 1.0, cp.tanh(kh) / cp.where(kh == 0, 1.0, kh))
    # Apply screening factor (0=short-circuit, 1=open-circuit)
    n_eff = n_kernel * screening_factor

    # In-plane Lifshitz gradient + 2D depol projector
    if use_elec:
        depol_xx = KX**2 / K2_safe
        depol_yy = KY**2 / K2_safe
        depol_xy = KX * KY / K2_safe
    else:
        depol_xx = depol_yy = depol_xy = 0.0

    M_xx = G11 * KX**2 + G44 * KY**2 + depol_xx / EPSILON_0
    M_yy = G44 * KX**2 + G11 * KY**2 + depol_yy / EPSILON_0
    M_xy = (G12 + G44) * KX * KY + depol_xy / EPSILON_0

    # Pz block: G44 * k_par^2 + thin-film depol
    M_zz = G44 * K2 + n_eff / EPSILON_0

    # Zero mode handling:
    # - In-plane: no gradient, no depol (uniform in-plane P has no divergence)
    M_xx[0, 0] = 0.0
    M_yy[0, 0] = 0.0
    M_xy[0, 0] = 0.0
    # - Pz: no gradient, but FULL depol for uniform Pz (this is the key open-circuit penalty)
    M_zz[0, 0] = screening_factor / EPSILON_0

    return KX, KY, K2_safe, K_par, n_eff, M_xx, M_yy, M_xy, M_zz, dx, dy

# ------------------- POISSON SOLVER (IN-PLANE) -------------------
def solve_poisson_inplane(Px, Py, KX, KY, K2_safe):
    """Solve nabla^2 Phi = div(P_inplane), return E_inplane = -grad(Phi).
    Note: Pz does NOT contribute (no z-derivatives in 2D), its depol is in M_zz.
    """
    Px_hat = cp.fft.fft2(Px)
    Py_hat = cp.fft.fft2(Py)
    div_P = 1j * (KX * Px_hat + KY * Py_hat) / EPSILON_0
    Phi_hat = -div_P / K2_safe
    Phi_hat[0, 0] = 0.0
    Ex = cp.fft.ifft2(-1j * KX * Phi_hat).real
    Ey = cp.fft.ifft2(-1j * KY * Phi_hat).real
    return Ex, Ey

# ------------------- LANDAU-DERIVONSHIRE BULK (3-COMPONENT) -------------------
def landau_deriv_3c(Px, Py, Pz):
    """d f_bulk / d P for 3-component tetragonal Landau-Devonshire, with optional
    substrate anisotropy β_sub*(Px²+Py²)/2 that favors Pz over in-plane P:
        f_bulk = a1*(Px^2+Py^2+Pz^2)
               + a11*(Px^4+Py^4+Pz^4)
               + a12*(Px^2*Py^2 + Py^2*Pz^2 + Px^2*Pz^2)
               + (β_sub/2)*(Px^2 + Py^2)        [substrate anisotropy]
    Returns (dF/dPx, dF/dPy, dF/dPz) -- NONLINEAR part, computed in real space.
    """
    Px2, Py2, Pz2 = Px**2, Py**2, Pz**2
    dPx = (2*alpha1*Px
           + 4*alpha11*Px*Px2
           + 2*alpha12*Px*(Py2 + Pz2)
           + 2*beta_sub*Px)
    dPy = (2*alpha1*Py
           + 4*alpha11*Py*Py2
           + 2*alpha12*Py*(Px2 + Pz2)
           + 2*beta_sub*Py)
    dPz = (2*alpha1*Pz
           + 4*alpha11*Pz*Pz2
           + 2*alpha12*Pz*(Px2 + Py2))
    return dPx, dPy, dPz

# ------------------- SIMULATION CORE -------------------
def run_simulation():
    cp.random.seed(seed)
    KX, KY, K2_safe, K_par, n_eff, M_xx, M_yy, M_xy, M_zz, dx, dy = build_k_operators(
        Nx, Ny, L_box, h_film
    )

    Ps_eq_gpu = cp.sqrt(-alpha1 / (2*alpha11)) if alpha1 < 0 and alpha11 > 0 else cp.asarray(1.0)
    ps_eq_float = float(to_cpu(Ps_eq_gpu))

    Px, Py, Pz = make_initial_P(Nx, Ny, L_box, init_mode, noise_amp, noise_cutoff, Ps_eq_gpu)

    snapshots, energy_log = [], []

    for step in range(n_steps + 1):
        # Nonlinear Landau derivative (real space)
        dF_bulk_x, dF_bulk_y, dF_bulk_z = landau_deriv_3c(Px, Py, Pz)

        # In-plane depol field (from Poisson, linear but treated as part of NL driving)
        if use_elec:
            Ex_dep, Ey_dep = solve_poisson_inplane(Px, Py, KX, KY, K2_safe)
            dF_elec_x = -Ex_dep
            dF_elec_y = -Ey_dep
        else:
            dF_elec_x = dF_elec_y = 0.0

        # Pz depol field is already in M_zz (linear operator), no real-space contribution
        dF_elec_z = 0.0

        NL_x = dF_bulk_x + dF_elec_x - Ex_ext
        NL_y = dF_bulk_y + dF_elec_y - Ey_ext
        NL_z = dF_bulk_z + dF_elec_z - Ez_ext

        # Fourier transforms
        Px_hat = cp.fft.fft2(Px)
        Py_hat = cp.fft.fft2(Py)
        Pz_hat = cp.fft.fft2(Pz)
        NLx_hat = cp.fft.fft2(NL_x)
        NLy_hat = cp.fft.fft2(NL_y)
        NLz_hat = cp.fft.fft2(NL_z)

        rhs_x = Px_hat - dt * Gamma * NLx_hat
        rhs_y = Py_hat - dt * Gamma * NLy_hat
        rhs_z = Pz_hat - dt * Gamma * NLz_hat

        # --- In-plane 2x2 solve ---
        A = 1.0 + dt * Gamma * M_xx
        D = 1.0 + dt * Gamma * M_yy
        B = dt * Gamma * M_xy
        det = A * D - B * B
        Px_hat_new = (D * rhs_x - B * rhs_y) / det
        Py_hat_new = (-B * rhs_x + A * rhs_y) / det

        # --- Pz 1x1 solve ---
        Pz_hat_new = rhs_z / (1.0 + dt * Gamma * M_zz)

        Px = cp.fft.ifft2(Px_hat_new).real
        Py = cp.fft.ifft2(Py_hat_new).real
        Pz = cp.fft.ifft2(Pz_hat_new).real

        # Diagnostics
        if step % save_every == 0 or step == n_steps:
            Px2, Py2, Pz2 = Px**2, Py**2, Pz**2
            f_bulk = (alpha1*(Px2+Py2+Pz2)
                      + alpha11*(Px2*Px2+Py2*Py2+Pz2*Pz2)
                      + alpha12*(Px2*Py2+Py2*Pz2+Px2*Pz2)
                      + 0.5*beta_sub*(Px2+Py2))
            Px_x, Px_y = cp.gradient(Px, dx, dy)
            Py_x, Py_y = cp.gradient(Py, dx, dy)
            Pz_x, Pz_y = cp.gradient(Pz, dx, dy)
            f_grad = (0.5*G11*(Px_x**2 + Py_y**2)
                      + G12*Px_x*Py_y
                      + 0.5*G44*((Px_y + Py_x)**2 + Pz_x**2 + Pz_y**2))
            # Pz depol energy (in Fourier)
            Pz_hat_for_E = cp.fft.fft2(Pz)
            f_depol_Pz = 0.5 * cp.sum(n_eff * cp.abs(Pz_hat_for_E)**2) / (Nx*Ny) / EPSILON_0 * (Nx*Ny) * dx * dy
            # Properly: f = (1/V) integral of (n_eff/(2*eps0)) |Pz_hat|^2 dk. Use mean:
            f_depol_Pz = cp.mean((n_eff / (2*EPSILON_0)) * cp.abs(Pz_hat_for_E)**2) / (Nx*Ny)
            # In-plane depol energy
            if use_elec:
                Ex_dep, Ey_dep = solve_poisson_inplane(Px, Py, KX, KY, K2_safe)
                f_depol_in = -0.5 * (Ex_dep*Px + Ey_dep*Py)
            else:
                f_depol_in = cp.zeros_like(Px)
            total = float(cp.mean(f_bulk + f_grad + f_depol_Pz + f_depol_in))
            energy_log.append((step, float(cp.mean(f_bulk)),
                               float(cp.mean(f_grad)),
                               float(cp.mean(f_depol_Pz)),
                               float(cp.mean(f_depol_in) if use_elec else 0.0),
                               total))
            snapshots.append((step, to_cpu(Px), to_cpu(Py), to_cpu(Pz)))

    return snapshots, energy_log, ps_eq_float

# ------------------- VISUALIZATION HELPERS -------------------
def polarization_color_3c(Px, Py, Pz, Ps):
    """HSV coloring: hue = in-plane direction, value = |P|/Ps."""
    mag = np.sqrt(Px**2 + Py**2 + Pz**2)
    angle = np.arctan2(Py, Px)
    hue = (angle + np.pi) / (2*np.pi)
    sat = np.clip(mag / (Ps + 1e-12), 0, 1)
    hsv = np.stack([hue, np.ones_like(hue)*0.9, sat], axis=-1)
    return hsv_to_rgb(hsv)

def overlay_quiver(ax, Px, Py, L, skip):
    Nx_, Ny_ = Px.shape
    x = np.linspace(0, L, Nx_)
    y = np.linspace(0, L, Ny_)
    X, Y = np.meshgrid(x, y, indexing='ij')
    ax.quiver(X[::skip,::skip], Y[::skip,::skip],
              Px[::skip,::skip], Py[::skip,::skip],
              color='white', scale=30, width=0.003, headwidth=4,
              headlength=5, pivot='middle')

# ------------------- RUN: store in session_state -------------------
if run_button:
    with st.spinner(f"Running FFT phase-field on {'GPU' if GPU_AVAILABLE else 'CPU'}..."):
        snaps, energy_log, ps_eq_float = run_simulation()
    st.session_state["snaps"] = snaps
    st.session_state["energy_log"] = energy_log
    st.session_state["ps_eq"] = ps_eq_float
    st.success(f"Done. {len(snaps)} frames saved. |Ps| ≈ {ps_eq_float*100:.2f} μC/cm²")

# ------------------- VISUALIZATION (outside run_button) -------------------
if "snaps" in st.session_state and st.session_state["snaps"]:
    snaps = st.session_state["snaps"]
    energy_log = st.session_state["energy_log"]
    ps_eq = st.session_state["ps_eq"]

    # Final snapshot
    st.header("Final polarization field")
    step_f, Px_f, Py_f, Pz_f = snaps[-1]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        vmax = max(ps_eq, np.abs(Pz_f).max())
        im = ax.imshow(Pz_f, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, L_box_nm, 0, L_box_nm])
        plt.colorbar(im, ax=ax, label="Pz (C/m²)")
        ax.set_title(f"Pz @ step {step_f}  (out-of-plane, 180° stripe domains)")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        if show_arrows:
            overlay_quiver(ax, Px_f, Py_f, L_box_nm, arrow_skip)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        rgb = polarization_color_3c(Px_f, Py_f, Pz_f, ps_eq)
        ax.imshow(rgb, origin='lower', extent=[0, L_box_nm, 0, L_box_nm])
        ax.set_title(f"|P| direction @ step {step_f}")
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        st.pyplot(fig)

    # In-plane components
    st.subheader("In-plane polarization components (Px, Py)")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(Px_f, origin='lower', cmap='RdBu_r',
                       vmin=-ps_eq*0.3, vmax=ps_eq*0.3,
                       extent=[0, L_box_nm, 0, L_box_nm])
        plt.colorbar(im, ax=ax, label="Px (C/m²)")
        ax.set_title(f"Px @ step {step_f}")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(Py_f, origin='lower', cmap='RdBu_r',
                       vmin=-ps_eq*0.3, vmax=ps_eq*0.3,
                       extent=[0, L_box_nm, 0, L_box_nm])
        plt.colorbar(im, ax=ax, label="Py (C/m²)")
        ax.set_title(f"Py @ step {step_f}")
        st.pyplot(fig)

    # Energy evolution
    st.header("Energy evolution")
    energy_log_arr = np.array(energy_log)
    steps = energy_log_arr[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, energy_log_arr[:, 1], label="f_bulk (Landau)")
    ax.plot(steps, energy_log_arr[:, 2], label="f_grad (Lifshitz)")
    ax.plot(steps, energy_log_arr[:, 3], label="f_depol_Pz (thin-film)")
    if use_elec:
        ax.plot(steps, energy_log_arr[:, 4], label="f_depol_inplane (2D projector)")
    ax.plot(steps, energy_log_arr[:, 5], 'k-', lw=2.5, label="Total f")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy density (J/m³)")
    ax.set_yscale("symlog", linthresh=1e3)
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Frame scrubber
    st.header("Time evolution playback")
    frame = st.slider("Frame", 0, len(snaps) - 1, len(snaps) - 1, key="frame_slider")
    s_i, Px_i, Py_i, Pz_i = snaps[frame]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Pz_i, origin='lower', cmap='RdBu_r',
                   vmin=-ps_eq, vmax=ps_eq,
                   extent=[0, L_box_nm, 0, L_box_nm])
    plt.colorbar(im, ax=ax, label="Pz (C/m²)")
    ax.set_title(f"Pz @ step {s_i}")
    if show_arrows:
        overlay_quiver(ax, Px_i, Py_i, L_box_nm, arrow_skip)
    st.pyplot(fig)

    # CSV export
    import pandas as pd
    df = pd.DataFrame(energy_log, columns=["step", "f_bulk", "f_grad",
                                            "f_depol_Pz", "f_depol_inplane", "f_total"])
    st.dataframe(df.head(20).style.format("{:.4e}"))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download energy CSV", csv, "ferro_energy.csv", "text/csv")

else:
    st.info("Configure parameters in the sidebar and click **Run simulation**.")

# ------------------- NOTES -------------------
st.markdown("""
---
**Key physics**

* **3-component P = (Px, Py, Pz)** with Pz as out-of-plane. This is the
  correct geometry for a thin film (film in xy-plane, Pz normal to film).

* **Thin-film depolarization kernel** for Pz:
    n(k) = tanh(k·h/2) / (k·h/2)
  with n(0)=1 (full depol for uniform Pz) and n(∞)→0 (no depol for fine stripes).
  This is the open-circuit boundary condition in z, captured exactly in Fourier space.

* **Screening factor** s ∈ [0,1]:
    n_eff(k) = s * n(k)
  s=1: open-circuit (no electrodes, full depol).
  s=0: short-circuit (perfect electrodes, no depol).
  Intermediate s ≈ 0.6–0.7 typically gives 180° stripe domain formation
  (uniform Pz penalized, stripes less penalized).

* **In-plane (Px, Py) depolarization** uses the 2D projector k⊗k/k²
  (volume bound charges from ∇·P_inplane).

* **Landau parameters**: for proper tetragonal symmetry, α₁₂ > 2α₁₁.
  Material presets: PbTiO3-like (|Ps|≈50 μC/cm², ξ≈0.6 nm) or
  HfO2-like (|Ps|≈14 μC/cm², ξ≈0.2 nm).

* **Why semi-implicit?** Linear part (gradient + depol) treated implicitly
  → unconditionally stable. Nonlinear Landau part treated explicitly.
  Time step set by `dt * Γ * |α₁| < 1`.

**Units**: all SI. P in C/m², E in V/m, length in m, time in s, energy in J/m³.
""")
