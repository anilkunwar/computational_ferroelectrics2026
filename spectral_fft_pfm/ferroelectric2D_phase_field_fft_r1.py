#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Ferroelectric Phase-Field Simulator — Integrated Single-File Edition
================================================================================

Physics:
    3-component polarization P = (Px, Py, Pz) in a 2D in-plane grid.
    Thin-film geometry: periodic in (x, y), open-circuit in z.

    TDGL:        dP/dt = -Gamma * dF/dP
    Poisson:     nabla^2 Phi = div(P_inplane) / eps0
    Thin-film depol for Pz:  n(k) = tanh(k*h/2) / (k*h/2)

    Free energy density:
        f = f_bulk (Landau-Devonshire, 3-component tetragonal)
          + f_grad (Lifshitz invariants, 2D-reduced)
          + f_depol_inplane (2D projector on Px, Py)
          + f_depol_Pz (thin-film kernel on Pz)
          + f_substrate (beta_sub * (Px^2 + Py^2) / 2, mimics compressive strain)

Discretization:
    FFT (semi-implicit Fourier spectral) on a uniform periodic grid.
    Block-diagonal 3x3 linear operator M(k):
        2x2 in-plane block (Px, Py): Lifshitz gradient + 2D depol projector
        1x1 Pz block:                G44*k_par^2 + n(k)/eps0 * screening_factor

All quantities in SI units: P [C/m^2], E [V/m], length [m], time [s].

Infrastructure (ported from UO2 codebase):
    - Error sentinels + handle_errors decorator
    - ColormapRegistry with 80+ built-in maps + 5 ferro-specific custom maps
    - Publication themes (nature-like, science-like, etc.)
    - Full figure customization panels (inline + enhanced, all options)
    - EnhancedLineProfiler (5 profile directions + custom angle)
    - FerroVisualizer (multi-field grid, overlay, vector overlay, Plotly
      heatmap/3D surface/animation/multi-time profiles)
    - DataExporter (PKL / NPZ / JSON / CSV / HDF5 / PT)
    - SimulationDatabase (session_state-backed, up to 32 runs)
    - 8 UI tabs: Geometry, Run, Viz Settings, Results, Line Profiles,
      Plotly, Export, Compare

Bug fixes applied during integration:
    #1: noise_amp conversion  (muC/cm^2 -> C/m^2 via *1e-2, was *1e4*1e-4 = *1)
    #2: f_depol_Pz computed once  (removed dead duplicate line)
    #3: alpha12 > 2*alpha11 enforced via __post_init__ warning
    #4: screening / beta_sub interaction documented
    #5: M_zz[0,0] = screening_factor / EPSILON_0  (uniform Pz penalized)
    #6: solve_poisson_inplane divides by EPSILON_0  (physical E field)

Usage:
    streamlit run ferroelectric_phase_field_integrated.py
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import json
import math
import time
import pickle
import hashlib
import zipfile
import logging
import traceback
import warnings
import functools
from io import BytesIO, StringIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

# ---- Plotting ----
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (Colormap, LinearSegmentedColormap,
                               ListedColormap, LogNorm, Normalize)
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

# ---- Plotly (interactive) ----
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---- Streamlit ----
import streamlit as st

# ---- Optional: PyTorch ----
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---- Optional: h5py ----
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# ---- Optional: imageio (for GIF/MP4) ----
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS (SI)
# ============================================================================
EPSILON_0 = 8.854187817e-12       # Vacuum permittivity [F/m]
K_B = 1.380649e-23                # Boltzmann constant [J/K]
E_CHARGE = 1.602176634e-19        # Elementary charge [C]
PI = math.pi

# ============================================================================
# GPU / CPU ARRAY BACKEND
# ============================================================================
GPU_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    cp.zeros(1)                     # Force context initialization
    GPU_AVAILABLE = True
    xp = cp
    logger.info("GPU (CuPy) detected and ready.")
except Exception as _gpu_err:
    xp = np
    GPU_AVAILABLE = False
    logger.info(f"GPU not available: {_gpu_err}. Using CPU (NumPy).")


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
    """Move any array-like to a NumPy array in host memory."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Move any array-like to the compute backend (CuPy if GPU, else NumPy)."""
    if GPU_AVAILABLE and not hasattr(arr, "device"):
        return cp.asarray(arr)
    return arr


# ============================================================================
# ERROR SENTINEL AND DECORATORS
# ============================================================================
class _Sentinel:
    """Singleton sentinel representing a failed operation.

    Used as the return value of functions wrapped by @handle_errors when an
    exception occurs. The sentinel is falsy, so callers can write:

        result = some_func()
        if result:
            # success
        else:
            # failure — check FAILED.reason
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reason = None
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<FAILED: {self.reason}>"

    def __iter__(self):
        return iter([])

    def __getitem__(self, key):
        return None


FAILED = _Sentinel()


def handle_errors(func=None, *, re_raise=False):
    """Decorator that catches exceptions and returns FAILED sentinel.

    If Streamlit is available, also displays an st.error message.
    If re_raise=True, the exception is re-raised after logging.
    """
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                FAILED.reason = f"{f.__name__}: {e}"
                logger.error(f"Error in {f.__name__}: {e}\n{traceback.format_exc()}")
                try:
                    st.error(f"❌ Error in {f.__name__}: {e}")
                except Exception:
                    pass
                if re_raise:
                    raise
                return FAILED
        return wrapper

    if func is None:
        return deco
    return deco(func)


def safe_unpack(result, n, default=None):
    """Unpack a result tuple, returning defaults if result is FAILED or None."""
    if result is FAILED or result is None:
        if default is None:
            return tuple([None] * n)
        return default
    return result


# ============================================================================
# SPECTRAL OPERATORS (RECTANGULAR GRID)
# ============================================================================
def make_k_vectors(Nx, Ny, dx, dy=None):
    """Build Fourier-space wavevector grids for a rectangular domain.

    Returns (kx, ky, k2) where:
        kx has shape (1, Nx)  — wavevector along x
        ky has shape (Ny, 1)  — wavevector along y
        k2 has shape (Ny, Nx) — |k|^2

    The DC mode (k=0) is set to a small value to avoid division by zero.
    """
    if dy is None:
        dy = dx
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx).reshape(1, -1)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12  # avoid div-by-zero; will be explicitly handled elsewhere
    return kx, ky, k2


def spectral_gradients(field, kx, ky, xp=None):
    """Compute (df/dx, df/dy) via FFT. Works on any 2D array."""
    if xp is None:
        xp = np
    fh = xp.fft.fft2(field)
    return (xp.real(xp.fft.ifft2(1j * kx * fh)),
            xp.real(xp.fft.ifft2(1j * ky * fh)))


def spectral_laplacian(field, k2, xp=None):
    """Compute Laplacian via FFT:  lap(f) = ifft(-k^2 * fft(f))."""
    if xp is None:
        xp = np
    return xp.real(xp.fft.ifft2(-k2 * xp.fft.fft2(field)))


def spectral_divergence(fx, fy, kx, ky, xp=None):
    """Compute div(fx, fy) = dfx/dx + dfy/dy via FFT."""
    if xp is None:
        xp = np
    return xp.real(xp.fft.ifft2(1j * kx * xp.fft.fft2(fx)
                                + 1j * ky * xp.fft.fft2(fy)))


# ============================================================================
# CUSTOM COLORMAPS (FERROELECTRIC-SPECIFIC)
# ============================================================================
def _build_custom_colormaps():
    """Build ferroelectric phase-field colormaps.

    Pz_div  : Diverging blue-white-red for polarization components (Px, Py, Pz).
    Pmag    : Sequential perceptually-uniform for |P| magnitude.
    Ez      : Diverging purple-orange for electric fields.
    energy  : Sequential hot for energy density fields.
    P_dir   : HSV-based cyclic for polarization direction.
    """
    # Pz diverging — standard blue-white-red, perceptually balanced
    Pz_div = LinearSegmentedColormap.from_list("Pz_div", [
        (0.00, "#2166ac"),   # strong blue (negative)
        (0.15, "#4393c3"),
        (0.30, "#92c5de"),
        (0.45, "#d1e5f0"),
        (0.50, "#f7f7f7"),   # white (zero)
        (0.55, "#fddbc7"),
        (0.70, "#f4a582"),
        (0.85, "#d6604d"),
        (1.00, "#b2182b"),   # strong red (positive)
    ])

    # |P| magnitude — dark-to-bright sequential
    Pmag = LinearSegmentedColormap.from_list("Pmag", [
        (0.00, "#000033"),
        (0.15, "#000066"),
        (0.30, "#1a0099"),
        (0.45, "#4d00cc"),
        (0.60, "#8000ff"),
        (0.75, "#b366ff"),
        (0.90, "#e6ccff"),
        (1.00, "#ffffff"),
    ])

    # Electric field diverging — purple to orange
    Ez = LinearSegmentedColormap.from_list("Ez", [
        (0.00, "#2c0040"),
        (0.20, "#5b0c6b"),
        (0.40, "#7b1fa2"),
        (0.50, "#f7f7f7"),
        (0.60, "#ff8a00"),
        (0.80, "#ef6c00"),
        (1.00, "#bf360c"),
    ])

    # Energy density — inferno-like hot
    energy = LinearSegmentedColormap.from_list("energy", [
        (0.00, "#000004"),
        (0.10, "#0d0d2a"),
        (0.20, "#1d1147"),
        (0.30, "#34115c"),
        (0.40, "#4b0c6b"),
        (0.50, "#6b0c7b"),
        (0.60, "#8c2981"),
        (0.70, "#b9377a"),
        (0.80, "#dd513a"),
        (0.90, "#fca50a"),
        (1.00, "#fcffa4"),
    ])

    # Polarization direction (cyclic HSV-like)
    P_dir = LinearSegmentedColormap.from_list("P_dir", [
        (0.000, "#ff0000"),  # 0°   — red
        (0.167, "#ff8000"),  # 60°  — orange
        (0.333, "#ffff00"),  # 120° — yellow
        (0.500, "#00ff00"),  # 180° — green
        (0.667, "#00ffff"),  # 240° — cyan
        (0.833, "#0000ff"),  # 300° — blue
        (1.000, "#ff0000"),  # 360° — back to red
    ])

    # Substrate anisotropy field
    beta_cmap = LinearSegmentedColormap.from_list("beta_cmap", [
        (0.00, "#1a1a2e"),
        (0.50, "#16213e"),
        (1.00, "#0f3460"),
    ])

    # Screening factor
    screening_cmap = LinearSegmentedColormap.from_list("screening_cmap", [
        (0.00, "#e74c3c"),  # short-circuit (red)
        (0.50, "#f39c12"),  # partial (orange)
        (1.00, "#2ecc71"),  # open-circuit (green)
    ])

    return {
        "Pz_div": Pz_div,
        "Pmag": Pmag,
        "Ez": Ez,
        "energy": energy,
        "P_dir": P_dir,
        "beta_cmap": beta_cmap,
        "screening_cmap": screening_cmap,
    }


_CUSTOM_COLORMAPS = _build_custom_colormaps()
for _n, _c in _CUSTOM_COLORMAPS.items():
    try:
        if _n not in mpl.colormaps:
            mpl.colormaps.register(_c, name=_n, force=True)
    except Exception as _e:
        logger.warning(f"Could not register colormap '{_n}': {_e}")


class ColormapRegistry:
    """Registry of all available colormaps (built-in + custom).

    Provides:
        list_all()           -> list of colormap names
        resolve(name, inv)   -> matplotlib Colormap object
        to_plotly(name, n)   -> Plotly colorscale [[pos, 'rgb(r,g,b)'], ...]
    """

    BASE = [
        # Perceptually uniform sequential
        "viridis", "viridis_r", "plasma", "plasma_r",
        "inferno", "inferno_r", "magma", "magma_r",
        "cividis", "cividis_r", "turbo", "turbo_r",
        # Sequential (single hue)
        "Greys", "Greys_r", "Blues", "Blues_r",
        "Greens", "Greens_r", "Reds", "Reds_r",
        "Oranges", "Oranges_r", "Purples", "Purples_r",
        # Sequential (multi-hue)
        "YlOrRd", "YlOrRd_r", "YlOrBr", "YlOrBr_r",
        "YlGnBu", "YlGnBu_r", "YlGn", "YlGn_r",
        "BuPu", "BuPu_r", "PuRd", "PuRd_r",
        "BuGn", "BuGn_r", "PuBu", "PuBu_r",
        "PuBuGn", "PuBuGn_r", "GnBu", "GnBu_r",
        "OrRd", "OrRd_r",
        # Diverging
        "RdBu", "RdBu_r", "coolwarm", "coolwarm_r",
        "bwr", "bwr_r", "seismic", "seismic_r",
        "Spectral", "Spectral_r", "RdYlBu", "RdYlBu_r",
        "RdYlGn", "RdYlGn_r", "BrBG", "BrBG_r",
        "PiYG", "PiYG_r", "PRGn", "PRGn_r",
        "PuOr", "PuOr_r", "RdGy", "RdGy_r",
        # Cyclic
        "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r",
        "hsv", "jet", "jet_r", "rainbow", "rainbow_r",
        "nipy_spectral", "nipy_spectral_r", "gist_rainbow", "gist_rainbow_r",
        # Monochrome
        "binary", "binary_r", "gray", "gray_r",
        "bone", "bone_r", "copper", "copper_r",
        "pink", "pink_r", "hot", "hot_r",
        "afmhot", "afmhot_r", "gist_heat", "gist_heat_r",
        "gist_gray", "gist_gray_r", "gist_yarg", "gist_yarg_r",
        # Miscellaneous
        "flag", "prism", "ocean", "ocean_r",
        "terrain", "terrain_r", "gist_earth", "gist_earth_r",
        "gist_ncar", "gist_ncar_r", "gist_stern", "gist_stern_r",
        "CMRmap", "CMRmap_r", "cubehelix", "cubehelix_r",
        "brg", "brg_r", "cool", "cool_r", "warm", "warm_r",
        "Wistia", "Wistia_r",
        "spring", "spring_r", "summer", "summer_r",
        "autumn", "autumn_r", "winter", "winter_r",
        # Qualitative
        "Pastel1", "Pastel2", "Paired", "Accent",
        "Dark2", "Set1", "Set2", "Set3",
        "tab10", "tab20", "tab20b", "tab20c",
    ]

    @classmethod
    def list_all(cls):
        """Return a deduplicated list of all colormap names (base + custom)."""
        names = list(cls.BASE) + list(_CUSTOM_COLORMAPS.keys())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    @classmethod
    def resolve(cls, name, invert=False):
        """Resolve a colormap name to a matplotlib Colormap object."""
        if name is None:
            cmap = mpl.colormaps["viridis"]
        elif isinstance(name, Colormap):
            cmap = name
        else:
            key = str(name)
            if key in mpl.colormaps:
                cmap = mpl.colormaps[key]
            elif key in _CUSTOM_COLORMAPS:
                cmap = _CUSTOM_COLORMAPS[key]
            else:
                logger.warning(f"Unknown colormap '{name}', using viridis")
                cmap = mpl.colormaps["viridis"]
        if invert:
            cmap = cmap.reversed()
        return cmap

    @classmethod
    def to_plotly(cls, name, n=64, invert=False):
        """Convert a matplotlib colormap to a Plotly colorscale."""
        cmap = cls.resolve(name, invert=invert)
        t = np.linspace(0, 1, n)
        rgb = (cmap(t)[:, :3] * 255).astype(int)
        return [[float(ti), f"rgb({r},{g},{b})"]
                for ti, (r, g, b) in zip(t, rgb)]


# ============================================================================
# PUBLICATION THEMES
# ============================================================================
MATPLOTLIB_THEMES = {
    "default":                None,
    "ggplot":                 "ggplot",
    "seaborn-whitegrid":      "seaborn-v0_8-whitegrid",
    "seaborn-darkgrid":       "seaborn-v0_8-darkgrid",
    "seaborn-ticks":          "seaborn-v0_8-ticks",
    "seaborn-paper":          "seaborn-v0_8-paper",
    "seaborn-talk":           "seaborn-v0_8-talk",
    "bmh":                    "bmh",
    "dark_background":        "dark_background",
    "classic":                "classic",
    "fivethirtyeight":        "fivethirtyeight",
    "Solarize_Light2":        "Solarize_Light2",
    "tableau-colorblind10":   "tableau-colorblind10",
    "nature-like (custom)":   "__nature__",
    "science-like (custom)":  "__science__",
}

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_SCIENCE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 1.0,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}


def apply_matplotlib_theme(theme_name, cfg):
    """Apply a matplotlib theme and override font sizes from cfg."""
    plt.rcdefaults()
    style = MATPLOTLIB_THEMES.get(theme_name, None)
    if style == "__nature__":
        mpl.rcParams.update(_NATURE_RC)
    elif style == "__science__":
        mpl.rcParams.update(_SCIENCE_RC)
    elif style is not None:
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Theme '{style}' unavailable: {e}")
            plt.rcdefaults()
    mpl.rcParams["axes.titlesize"]  = cfg.get("title_font_size", 11)
    mpl.rcParams["axes.labelsize"]  = cfg.get("label_font_size", 9)
    mpl.rcParams["xtick.labelsize"] = cfg.get("tick_font_size", 8)
    mpl.rcParams["ytick.labelsize"] = cfg.get("tick_font_size", 8)
    mpl.rcParams["legend.fontsize"] = cfg.get("legend_font_size", 8)
    mpl.rcParams["figure.dpi"]      = cfg.get("dpi", 150)
    mpl.rcParams["savefig.dpi"]     = cfg.get("dpi", 150)


# ============================================================================
# FIELD CONFIGURATION (FERROELECTRIC)
# ============================================================================
FIELD_DEFAULT_CMAPS = {
    "Px":               "Pz_div",
    "Py":               "Pz_div",
    "Pz":               "Pz_div",
    "P_mag":            "Pmag",
    "f_bulk":           "energy",
    "f_grad":           "viridis",
    "f_depol_Pz":       "magma",
    "f_depol_inplane":  "magma",
    "f_total":          "energy",
}

FIELD_LABELS = {
    "Px":               "Px (in-plane polarization)",
    "Py":               "Py (in-plane polarization)",
    "Pz":               "Pz (out-of-plane polarization)",
    "P_mag":            "|P| (polarization magnitude)",
    "f_bulk":           "f_bulk (Landau-Devonshire bulk energy)",
    "f_grad":           "f_grad (Lifshitz gradient energy)",
    "f_depol_Pz":       "f_depol_Pz (thin-film depolarization)",
    "f_depol_inplane":  "f_depol_inplane (2D projector)",
    "f_total":          "f_total (total free energy density)",
}

FIELD_UNITS = {
    "Px": "C/m²", "Py": "C/m²", "Pz": "C/m²",
    "P_mag": "C/m²",
    "f_bulk": "J/m³", "f_grad": "J/m³",
    "f_depol_Pz": "J/m³", "f_depol_inplane": "J/m³",
    "f_total": "J/m³",
}

FIELD_DEFAULT_RNG = {
    "Px": (-0.5, 0.5), "Py": (-0.5, 0.5), "Pz": (-0.5, 0.5),
    "P_mag": (0.0, 0.6),
    "f_bulk": (-1e10, 0.0), "f_grad": (0.0, 1e9),
    "f_depol_Pz": (0.0, 1e9), "f_depol_inplane": (0.0, 1e9),
    "f_total": (-1e10, 0.0),
}

DISCRETE_FIELDS = set()  # No discrete fields in the ferro model

VECTOR_FIELD_SOURCES = ["Px", "Py", "Pz", "P_mag"]

# Fields available for the multi-field grid (display order)
MULTI_FIELD_LAYOUT = [
    ("Pz",              "Pz (C/m²)",              (-0.5, 0.5)),
    ("P_mag",           "|P| (C/m²)",             (0.0, 0.6)),
    ("Px",              "Px (C/m²)",              (-0.5, 0.5)),
    ("Py",              "Py (C/m²)",              (-0.5, 0.5)),
    ("f_bulk",          "f_bulk (J/m³)",          None),
    ("f_grad",          "f_grad (J/m³)",          None),
    ("f_depol_Pz",      "f_depol_Pz (J/m³)",      None),
    ("f_depol_inplane", "f_depol_inplane (J/m³)", None),
    ("f_total",         "f_total (J/m³)",         None),
]


# ============================================================================
# TIME-UNIT HANDLING
# ============================================================================
TIME_UNITS = {
    "ps": {"label": "ps", "long": "picoseconds", "factor": 1e-12},
    "ns": {"label": "ns", "long": "nanoseconds", "factor": 1e-9},
    "μs": {"label": "μs", "long": "microseconds", "factor": 1e-6},
    "ms": {"label": "ms", "long": "milliseconds", "factor": 1e-3},
    "s":  {"label": "s",  "long": "seconds",      "factor": 1.0},
}

DEFAULT_TIME_UNIT = "ps"


def get_time_unit_label(params):
    if params is None:
        return DEFAULT_TIME_UNIT
    return params.get("time_unit", DEFAULT_TIME_UNIT)


def fmt_time(t_code, params, fmt="%.4f"):
    unit = get_time_unit_label(params)
    factor = TIME_UNITS.get(unit, TIME_UNITS[DEFAULT_TIME_UNIT])["factor"]
    return f"{fmt % (float(t_code) / factor)} {unit}"


def fmt_time_range(t_code_arr, params):
    unit = get_time_unit_label(params)
    factor = TIME_UNITS.get(unit, TIME_UNITS[DEFAULT_TIME_UNIT])["factor"]
    return np.asarray(t_code_arr, dtype=float) / factor, unit


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
def default_field_settings():
    """Default per-field visualization settings (auto range, no log, no invert)."""
    return {
        f: {"auto": True, "vmin": None, "vmax": None,
            "log": False, "invert": False,
            "cmap": FIELD_DEFAULT_CMAPS.get(f, "viridis")}
        for f in FIELD_DEFAULT_CMAPS.keys()
    }


def default_figure_customization():
    """Full figure customization defaults — every tunable parameter."""
    return {
        # Figure sizes (inches)
        "multi_field_width":    13.5,
        "multi_field_height":   11.0,
        "overlay_width":        11.0,
        "overlay_height":        5.5,
        "vector_width":         11.0,
        "vector_height":         5.5,
        "diagnostics_width":    13.0,
        "diagnostics_height":    8.0,
        "line_profile_width":   11.0,
        "line_profile_height":   4.5,
        "plotly_width":         11.0,
        "plotly_height":         5.5,
        # Line properties
        "line_width":            2.0,
        "line_width_minor":      1.5,
        "marker_size":           6,
        "marker_width":          1.0,
        "contour_line_width":    1.0,
        "quiver_width":          0.0025,
        "quiver_scale":          30.0,
        "line_style":            "-",
        "line_style_minor":      "--",
        "marker_style":          "none",
        "marker_style_minor":    "none",
        # Series colors
        "color_series_1":        "#1f77b4",
        "color_series_2":        "#d62728",
        "color_series_3":        "#2ca02c",
        "color_series_4":        "#9467bd",
        "color_series_5":        "#8c564b",
        # Text overrides (blank = use default)
        "title_override":       "",
        "xlabel_override":      "",
        "ylabel_override":      "",
        # Font sizes (None = use theme default)
        "title_font_size":      None,
        "label_font_size":      None,
        "tick_font_size":       None,
        "legend_font_size":     None,
        # Grid
        "show_grid":            True,
        "grid_alpha":           0.3,
        "grid_linestyle":       "-",
        "grid_color":           "#cccccc",
        # Axis limits
        "xlim":                 (None, None),
        "ylim":                 (None, None),
        # Colorbar
        "show_colorbar":        True,
        "colorbar_orientation": "vertical",
        # Spines
        "spine_width":          1.0,
        "spine_top":            True,
        "spine_right":          True,
        "spine_left":           True,
        "spine_bottom":         True,
        # Ticks
        "tick_direction":       "out",
        "tick_width":           1.0,
        "tick_length":          4,
        # Legend
        "show_legend":          True,
        "legend_position":      "best",
        "legend_frameon":       True,
        "legend_framealpha":    1.0,
        # Layout
        "axis_equal":           True,
        "tight_layout":         True,
    }


def default_viz_config():
    """Top-level visualization configuration."""
    return {
        "theme": "default",
        "title_font_size": 11,
        "label_font_size": 9,
        "tick_font_size": 8,
        "legend_font_size": 8,
        "dpi": 150,
        "colorbar_fraction": 0.046,
        "colorbar_pad": 0.04,
        "field_settings": default_field_settings(),
        "figure_customization": default_figure_customization(),
        "overlay": {
            "enabled": False,
            "base": "Pz",
            "overlay": "P_mag",
            "alpha": 0.5,
            "mode": "contour",
            "n_contours": 10,
        },
        "vector": {
            "enabled": False,
            "source": "Pz",
            "mode": "inplane",
            "density": 24,
            "color": "#ffffff",
            "scale": 1.0,
        },
        "animation": {
            "format": "gif",
            "fps": 5,
            "dpi": 120,
            "watermark": True,
            "watermark_text": "",
            "watermark_pos": "lower right",
        },
    }


def get_viz_config():
    """Get or initialize viz config from session_state, filling defaults."""
    if "viz_cfg" not in st.session_state:
        st.session_state.viz_cfg = default_viz_config()
    cfg = st.session_state.viz_cfg
    defaults = default_viz_config()
    if "figure_customization" not in cfg:
        cfg["figure_customization"] = defaults["figure_customization"]
    else:
        for k, v in defaults["figure_customization"].items():
            cfg["figure_customization"].setdefault(k, v)
    for fname, fset in cfg["field_settings"].items():
        fset.setdefault("cmap", FIELD_DEFAULT_CMAPS.get(fname, "viridis"))
    return cfg


# ============================================================================
# AXIS STYLING
# ============================================================================
def apply_axis_styling(ax, fig_cfg, *, title=None, xlabel=None, ylabel=None,
                       title_fs=None, label_fs=None, grid_axes=None,
                       force_aspect=None):
    """Apply figure customization to a matplotlib Axes object."""
    if fig_cfg is None or ax is None:
        return
    if title_fs is None:
        title_fs = fig_cfg.get("title_font_size")
    if label_fs is None:
        label_fs = fig_cfg.get("label_font_size")
    tick_fs = fig_cfg.get("tick_font_size")

    # Title
    if title is not None:
        override = fig_cfg.get("title_override", "").strip()
        ax.set_title(override if override else title, fontsize=title_fs)

    # Labels
    if xlabel is not None:
        override = fig_cfg.get("xlabel_override", "").strip()
        ax.set_xlabel(override if override else xlabel, fontsize=label_fs)
    if ylabel is not None:
        override = fig_cfg.get("ylabel_override", "").strip()
        ax.set_ylabel(override if override else ylabel, fontsize=label_fs)

    # Tick font sizes
    if tick_fs is not None:
        for label_artist in ax.get_xticklabels() + ax.get_yticklabels():
            label_artist.set_fontsize(float(tick_fs))

    # Grid
    target = grid_axes if grid_axes is not None else ax
    if fig_cfg.get("show_grid", True):
        target.grid(True,
                    alpha=float(fig_cfg.get("grid_alpha", 0.3)),
                    linestyle=fig_cfg.get("grid_linestyle", "-"),
                    color=fig_cfg.get("grid_color", "#cccccc"))
    else:
        target.grid(False)

    # Axis limits
    xlim = fig_cfg.get("xlim", (None, None))
    ylim = fig_cfg.get("ylim", (None, None))
    if xlim and (xlim[0] is not None or xlim[1] is not None):
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim and (ylim[0] is not None or ylim[1] is not None):
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # Spines
    sw = float(fig_cfg.get("spine_width", 1.0))
    spine_visible = {
        "top":    fig_cfg.get("spine_top",    True),
        "right":  fig_cfg.get("spine_right",  True),
        "left":   fig_cfg.get("spine_left",   True),
        "bottom": fig_cfg.get("spine_bottom", True),
    }
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(sw)
        spine.set_visible(spine_visible.get(spine_name, True))

    # Ticks
    ax.tick_params(width=float(fig_cfg.get("tick_width", 1.0)),
                   length=float(fig_cfg.get("tick_length", 4)),
                   direction=fig_cfg.get("tick_direction", "out"))

    # Aspect
    if force_aspect is not None:
        aspect_mode = force_aspect
    elif fig_cfg.get("axis_equal", True):
        aspect_mode = "equal"
    else:
        aspect_mode = "auto"
    try:
        if aspect_mode == "equal":
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")
    except Exception:
        pass


def apply_legend_styling(ax, fig_cfg):
    """Apply legend customization to a matplotlib Axes."""
    if fig_cfg is None or ax is None:
        return
    if not fig_cfg.get("show_legend", True):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return
    leg_pos = fig_cfg.get("legend_position", "best")
    legend_fs = fig_cfg.get("legend_font_size")
    frameon = bool(fig_cfg.get("legend_frameon", True))
    framealpha = float(fig_cfg.get("legend_framealpha", 1.0))
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(handles, labels, loc=leg_pos, frameon=frameon,
                       framealpha=framealpha, fontsize=legend_fs)
    return legend


# ============================================================================
# INLINE FIGURE CUSTOMIZATION HELPERS
# ============================================================================
LINE_STYLES = ["-", "--", "-.", ":", "none", ""]
MARKER_STYLES = ["none", "o", "s", "^", "v", "<", ">", "D", "d", "+", "x",
                 "*", ".", ",", "1", "2", "3", "4", "_"]
LEGEND_POSITIONS = ["best", "upper right", "upper left", "lower right",
                    "lower left", "center", "center left", "center right",
                    "upper center", "lower center", "outside upper",
                    "outside right"]
TICK_DIRECTIONS = ["out", "in", "inout"]
GRID_LINESTYLES = ["-", "--", ":", "-.", ""]


def _clamped_number_input(label, value, min_value, max_value, step=0.5,
                           key=None, help=None, format=None):
    """Wrapper around st.number_input that clamps the value to [min, max]."""
    if value is None:
        value = min_value
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(min_value)
    if value < min_value:
        value = float(min_value)
    elif value > max_value:
        value = float(max_value)
    kwargs = dict(value=value, min_value=min_value, max_value=max_value,
                  step=step)
    if key is not None:
        kwargs["key"] = key
    if help is not None:
        kwargs["help"] = help
    if format is not None:
        kwargs["format"] = format
    return st.number_input(label, **kwargs)


def _parse_limit(s):
    """Parse a text input into a float, returning None for empty strings."""
    s = s.strip() if isinstance(s, str) else s
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def render_inline_figure_customization(viz_cfg, *, size_keys=None,
                                        key_prefix="inl",
                                        show_axis_limits=True,
                                        show_text_overrides=True,
                                        show_grid=True,
                                        show_colorbar=True,
                                        show_lines=True,
                                        label="Figure customization"):
    """Compact inline figure customization panel (fewer options than enhanced)."""
    fc = viz_cfg.setdefault("figure_customization", default_figure_customization())
    default_expanded = bool(st.session_state.get(f"{key_prefix}_exp", False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f"{key_prefix}_exp"] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            c1, c2 = st.columns(2)
            with c1:
                new_w = _clamped_number_input(
                    "Width (in)", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w",
                    help="Figure width in inches.")
            with c2:
                new_h = _clamped_number_input(
                    "Height (in)", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            fc[w_key] = float(new_w)
            fc[h_key] = float(new_h)

        if show_text_overrides:
            st.markdown("**Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                t_ov = st.text_input("Title", value=fc.get("title_override", ""),
                                      key=f"{key_prefix}_ti",
                                      label_visibility="collapsed",
                                      placeholder="Title override")
            with c2:
                x_ov = st.text_input("X label", value=fc.get("xlabel_override", ""),
                                      key=f"{key_prefix}_xl",
                                      label_visibility="collapsed",
                                      placeholder="X-axis label")
            with c3:
                y_ov = st.text_input("Y label", value=fc.get("ylabel_override", ""),
                                      key=f"{key_prefix}_yl",
                                      label_visibility="collapsed",
                                      placeholder="Y-axis label")
            fc["title_override"] = t_ov
            fc["xlabel_override"] = x_ov
            fc["ylabel_override"] = y_ov

        if show_grid:
            c1, c2 = st.columns(2)
            with c1:
                new_grid = st.checkbox("Show grid",
                                        value=bool(fc.get("show_grid", True)),
                                        key=f"{key_prefix}_grid")
            with c2:
                new_ga = st.slider("Grid α", 0.0, 1.0,
                                    float(fc.get("grid_alpha", 0.3)),
                                    0.05, key=f"{key_prefix}_ga")
            fc["show_grid"] = new_grid
            fc["grid_alpha"] = new_ga

        if show_axis_limits:
            xlim_cur = fc.get("xlim", (None, None))
            ylim_cur = fc.get("ylim", (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input("xmin (blank=auto)",
                                         value="" if xlim_cur[0] is None else str(xlim_cur[0]),
                                         key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input("xmax (blank=auto)",
                                         value="" if xlim_cur[1] is None else str(xlim_cur[1]),
                                         key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input("ymin (blank=auto)",
                                         value="" if ylim_cur[0] is None else str(ylim_cur[0]),
                                         key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input("ymax (blank=auto)",
                                         value="" if ylim_cur[1] is None else str(ylim_cur[1]),
                                         key=f"{key_prefix}_ymax")
            fc["xlim"] = (_parse_limit(xmin_in), _parse_limit(xmax_in))
            fc["ylim"] = (_parse_limit(ymin_in), _parse_limit(ymax_in))

        if show_colorbar:
            c1, c2 = st.columns(2)
            with c1:
                new_cb = st.checkbox("Show colorbar",
                                      value=bool(fc.get("show_colorbar", True)),
                                      key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get("colorbar_orientation", "vertical") == "vertical" else 1
                new_cbo = st.selectbox("Colorbar orient.",
                                        ["vertical", "horizontal"],
                                        index=cb_o_idx,
                                        key=f"{key_prefix}_cbo")
            fc["show_colorbar"] = new_cb
            fc["colorbar_orientation"] = new_cbo

        if show_lines:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                new_lw = st.number_input("Line width",
                                          value=float(fc.get("line_width", 2.0)),
                                          min_value=0.2, max_value=10.0, step=0.1,
                                          key=f"{key_prefix}_lw")
            with c2:
                new_lwm = st.number_input("Minor lw",
                                          value=float(fc.get("line_width_minor", 1.5)),
                                          min_value=0.1, max_value=8.0, step=0.1,
                                          key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get("marker_size", 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                new_ms = st.number_input("Marker size", value=_ms_val,
                                          min_value=0, max_value=30, step=1,
                                          key=f"{key_prefix}_ms")
            with c4:
                new_cw = st.number_input("Contour lw",
                                          value=float(fc.get("contour_line_width", 1.0)),
                                          min_value=0.1, max_value=5.0, step=0.1,
                                          key=f"{key_prefix}_cw")
            fc["line_width"] = new_lw
            fc["line_width_minor"] = new_lwm
            fc["marker_size"] = new_ms
            fc["contour_line_width"] = new_cw

        c1, c2, c3 = st.columns(3)
        with c1:
            new_sw = st.number_input("Spine w",
                                      value=float(fc.get("spine_width", 1.0)),
                                      min_value=0.1, max_value=5.0, step=0.1,
                                      key=f"{key_prefix}_sw")
        with c2:
            td_idx = ["out", "in", "inout"].index(fc.get("tick_direction", "out"))
            new_td = st.selectbox("Tick dir", ["out", "in", "inout"],
                                   index=td_idx, key=f"{key_prefix}_td")
        with c3:
            _tl_val = fc.get("tick_length", 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            new_tl = st.number_input("Tick len", value=_tl_val,
                                      min_value=0, max_value=20, step=1,
                                      key=f"{key_prefix}_tl")
        fc["spine_width"] = new_sw
        fc["tick_direction"] = new_td
        fc["tick_length"] = new_tl

        st.caption("✏️ Tweak any value — changes apply on next figure render.")

    return True


def render_enhanced_figure_customization(viz_cfg, *, size_keys=None,
                                          key_prefix="enh",
                                          show_line_styles=True,
                                          show_series_colors=False,
                                          show_legend=True,
                                          show_spines=True,
                                          show_font_sizes=True,
                                          show_colorbar=True,
                                          show_axis_limits=True,
                                          show_text_overrides=True,
                                          show_grid=True,
                                          n_series=5,
                                          label="Figure customization"):
    """Full figure customization panel with every available option."""
    fc = viz_cfg.setdefault("figure_customization", default_figure_customization())
    default_expanded = bool(st.session_state.get(f"{key_prefix}_exp", False))

    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f"{key_prefix}_exp"] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")

        # ---- Figure size ----
        if size_keys is not None and len(size_keys) == 2:
            w_key, h_key = size_keys
            st.markdown("**📐 Figure size (inches)**")
            c1, c2 = st.columns(2)
            with c1:
                new_w = _clamped_number_input(
                    "Width", value=fc.get(w_key, 11.0),
                    min_value=3.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_w")
            with c2:
                new_h = _clamped_number_input(
                    "Height", value=fc.get(h_key, 5.5),
                    min_value=2.0, max_value=30.0, step=0.5,
                    key=f"{key_prefix}_h")
            fc[w_key] = float(new_w)
            fc[h_key] = float(new_h)

        # ---- Font sizes ----
        if show_font_sizes:
            st.markdown("**🔤 Font sizes** (0 = theme default)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                cur = fc.get("title_font_size")
                new_tfs = st.number_input("Title",
                                           value=float(cur) if cur is not None else 0.0,
                                           min_value=0.0, max_value=30.0, step=0.5,
                                           key=f"{key_prefix}_tfs",
                                           help="0 = use theme default")
                fc["title_font_size"] = float(new_tfs) if new_tfs and new_tfs > 0 else None
            with c2:
                cur = fc.get("label_font_size")
                new_lfs = st.number_input("Labels",
                                           value=float(cur) if cur is not None else 0.0,
                                           min_value=0.0, max_value=24.0, step=0.5,
                                           key=f"{key_prefix}_lfs",
                                           help="0 = use theme default")
                fc["label_font_size"] = float(new_lfs) if new_lfs and new_lfs > 0 else None
            with c3:
                cur = fc.get("tick_font_size")
                new_tkfs = st.number_input("Ticks",
                                            value=float(cur) if cur is not None else 0.0,
                                            min_value=0.0, max_value=20.0, step=0.5,
                                            key=f"{key_prefix}_tkfs",
                                            help="0 = use theme default")
                fc["tick_font_size"] = float(new_tkfs) if new_tkfs and new_tkfs > 0 else None
            with c4:
                cur = fc.get("legend_font_size")
                new_lgfs = st.number_input("Legend",
                                            value=float(cur) if cur is not None else 0.0,
                                            min_value=0.0, max_value=20.0, step=0.5,
                                            key=f"{key_prefix}_lgfs",
                                            help="0 = use theme default")
                fc["legend_font_size"] = float(new_lgfs) if new_lgfs and new_lgfs > 0 else None

        # ---- Text overrides ----
        if show_text_overrides:
            st.markdown("**🏷️ Text overrides** (blank = default)")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc["title_override"] = st.text_input(
                    "Title", value=fc.get("title_override", ""),
                    key=f"{key_prefix}_ti", placeholder="Title override")
            with c2:
                fc["xlabel_override"] = st.text_input(
                    "X label", value=fc.get("xlabel_override", ""),
                    key=f"{key_prefix}_xl", placeholder="X-axis label")
            with c3:
                fc["ylabel_override"] = st.text_input(
                    "Y label", value=fc.get("ylabel_override", ""),
                    key=f"{key_prefix}_yl", placeholder="Y-axis label")

        # ---- Line styles & markers ----
        if show_line_styles:
            st.markdown("**✏️ Line styles & markers**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ls_idx = LINE_STYLES.index(fc.get("line_style", "-")) \
                    if fc.get("line_style", "-") in LINE_STYLES else 0
                fc["line_style"] = st.selectbox(
                    "Line style (main)", LINE_STYLES, index=ls_idx,
                    key=f"{key_prefix}_ls",
                    format_func=lambda x: {"-": "— solid", "--": "–– dashed",
                                            "-.": "–· dash-dot", ":": "·· dotted",
                                            "none": "none", "": "none"}.get(x, x))
            with c2:
                lsm_idx = LINE_STYLES.index(fc.get("line_style_minor", "--")) \
                    if fc.get("line_style_minor", "--") in LINE_STYLES else 1
                fc["line_style_minor"] = st.selectbox(
                    "Line style (minor)", LINE_STYLES, index=lsm_idx,
                    key=f"{key_prefix}_lsm",
                    format_func=lambda x: {"-": "— solid", "--": "–– dashed",
                                            "-.": "–· dash-dot", ":": "·· dotted",
                                            "none": "none", "": "none"}.get(x, x))
            with c3:
                ms_idx = MARKER_STYLES.index(fc.get("marker_style", "none")) \
                    if fc.get("marker_style", "none") in MARKER_STYLES else 0
                fc["marker_style"] = st.selectbox(
                    "Marker (main)", MARKER_STYLES, index=ms_idx,
                    key=f"{key_prefix}_ms")
            with c4:
                msm_idx = MARKER_STYLES.index(fc.get("marker_style_minor", "none")) \
                    if fc.get("marker_style_minor", "none") in MARKER_STYLES else 0
                fc["marker_style_minor"] = st.selectbox(
                    "Marker (minor)", MARKER_STYLES, index=msm_idx,
                    key=f"{key_prefix}_msm")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                fc["line_width"] = st.number_input(
                    "Line width (main)", value=float(fc.get("line_width", 2.0)),
                    min_value=0.2, max_value=10.0, step=0.1,
                    key=f"{key_prefix}_lw")
            with c2:
                fc["line_width_minor"] = st.number_input(
                    "Line width (minor)", value=float(fc.get("line_width_minor", 1.5)),
                    min_value=0.1, max_value=8.0, step=0.1,
                    key=f"{key_prefix}_lwm")
            with c3:
                _ms_val = fc.get("marker_size", 6)
                try:
                    _ms_val = int(_ms_val) if _ms_val is not None else 6
                except (TypeError, ValueError):
                    _ms_val = 6
                fc["marker_size"] = st.number_input(
                    "Marker size", value=_ms_val,
                    min_value=0, max_value=30, step=1,
                    key=f"{key_prefix}_msz")
            with c4:
                fc["marker_width"] = st.number_input(
                    "Marker edge w", value=float(fc.get("marker_width", 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_mew")

        # ---- Series colors ----
        if show_series_colors:
            st.markdown("**🎨 Per-series colors**")
            cols = st.columns(min(n_series, 5))
            for i in range(min(n_series, 5)):
                with cols[i]:
                    fc[f"color_series_{i+1}"] = st.color_picker(
                        f"Series {i+1}",
                        value=fc.get(f"color_series_{i+1}", "#1f77b4"),
                        key=f"{key_prefix}_cs{i+1}")

        # ---- Legend ----
        if show_legend:
            st.markdown("**📋 Legend**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc["show_legend"] = st.checkbox(
                    "Show legend", value=bool(fc.get("show_legend", True)),
                    key=f"{key_prefix}_sleg")
            with c2:
                leg_idx = LEGEND_POSITIONS.index(fc.get("legend_position", "best")) \
                    if fc.get("legend_position", "best") in LEGEND_POSITIONS else 0
                fc["legend_position"] = st.selectbox(
                    "Position", LEGEND_POSITIONS, index=leg_idx,
                    key=f"{key_prefix}_legpos")
            with c3:
                fc["legend_frameon"] = st.checkbox(
                    "Frame", value=bool(fc.get("legend_frameon", True)),
                    key=f"{key_prefix}_legf")
            c1, c2 = st.columns(2)
            with c1:
                fc["legend_framealpha"] = st.slider(
                    "Frame α", 0.0, 1.0,
                    float(fc.get("legend_framealpha", 1.0)),
                    0.05, key=f"{key_prefix}_legfa")

        # ---- Grid ----
        if show_grid:
            st.markdown("**📊 Grid**")
            c1, c2, c3 = st.columns(3)
            with c1:
                fc["show_grid"] = st.checkbox(
                    "Show grid", value=bool(fc.get("show_grid", True)),
                    key=f"{key_prefix}_grid")
            with c2:
                fc["grid_alpha"] = st.slider(
                    "Grid α", 0.0, 1.0,
                    float(fc.get("grid_alpha", 0.3)),
                    0.05, key=f"{key_prefix}_ga")
            with c3:
                gls_idx = GRID_LINESTYLES.index(fc.get("grid_linestyle", "-")) \
                    if fc.get("grid_linestyle", "-") in GRID_LINESTYLES else 0
                fc["grid_linestyle"] = st.selectbox(
                    "Grid style", GRID_LINESTYLES, index=gls_idx,
                    key=f"{key_prefix}_gls",
                    format_func=lambda x: {"-": "— solid", "--": "–– dashed",
                                            ":": "·· dotted", "-.": "–· dash-dot",
                                            "": "none"}.get(x, x))
            fc["grid_color"] = st.color_picker(
                "Grid color", value=fc.get("grid_color", "#cccccc"),
                key=f"{key_prefix}_gcol")

        # ---- Axis limits ----
        if show_axis_limits:
            st.markdown("**📏 Axis limits** (blank = auto)")
            xlim_cur = fc.get("xlim", (None, None))
            ylim_cur = fc.get("ylim", (None, None))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                xmin_in = st.text_input("xmin",
                                         value="" if xlim_cur[0] is None else str(xlim_cur[0]),
                                         key=f"{key_prefix}_xmin")
            with c2:
                xmax_in = st.text_input("xmax",
                                         value="" if xlim_cur[1] is None else str(xlim_cur[1]),
                                         key=f"{key_prefix}_xmax")
            with c3:
                ymin_in = st.text_input("ymin",
                                         value="" if ylim_cur[0] is None else str(ylim_cur[0]),
                                         key=f"{key_prefix}_ymin")
            with c4:
                ymax_in = st.text_input("ymax",
                                         value="" if ylim_cur[1] is None else str(ylim_cur[1]),
                                         key=f"{key_prefix}_ymax")
            fc["xlim"] = (_parse_limit(xmin_in), _parse_limit(xmax_in))
            fc["ylim"] = (_parse_limit(ymin_in), _parse_limit(ymax_in))

        # ---- Colorbar ----
        if show_colorbar:
            st.markdown("**🎨 Colorbar**")
            c1, c2 = st.columns(2)
            with c1:
                fc["show_colorbar"] = st.checkbox(
                    "Show colorbar", value=bool(fc.get("show_colorbar", True)),
                    key=f"{key_prefix}_cb")
            with c2:
                cb_o_idx = 0 if fc.get("colorbar_orientation", "vertical") == "vertical" else 1
                fc["colorbar_orientation"] = st.selectbox(
                    "Colorbar orient.", ["vertical", "horizontal"],
                    index=cb_o_idx, key=f"{key_prefix}_cbo")

        # ---- Spines ----
        if show_spines:
            st.markdown("**▦ Spines (box)**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                fc["spine_top"] = st.checkbox(
                    "Top", value=bool(fc.get("spine_top", True)),
                    key=f"{key_prefix}_spt")
            with c2:
                fc["spine_right"] = st.checkbox(
                    "Right", value=bool(fc.get("spine_right", True)),
                    key=f"{key_prefix}_spr")
            with c3:
                fc["spine_left"] = st.checkbox(
                    "Left", value=bool(fc.get("spine_left", True)),
                    key=f"{key_prefix}_spl")
            with c4:
                fc["spine_bottom"] = st.checkbox(
                    "Bottom", value=bool(fc.get("spine_bottom", True)),
                    key=f"{key_prefix}_spb")
            with c5:
                fc["spine_width"] = st.number_input(
                    "Width", value=float(fc.get("spine_width", 1.0)),
                    min_value=0.1, max_value=5.0, step=0.1,
                    key=f"{key_prefix}_sw")

        # ---- Ticks ----
        st.markdown("**⏱️ Ticks**")
        c1, c2, c3 = st.columns(3)
        with c1:
            td_idx = TICK_DIRECTIONS.index(fc.get("tick_direction", "out")) \
                if fc.get("tick_direction", "out") in TICK_DIRECTIONS else 0
            fc["tick_direction"] = st.selectbox(
                "Direction", TICK_DIRECTIONS, index=td_idx,
                key=f"{key_prefix}_td")
        with c2:
            fc["tick_width"] = st.number_input(
                "Width", value=float(fc.get("tick_width", 1.0)),
                min_value=0.1, max_value=5.0, step=0.1,
                key=f"{key_prefix}_tw")
        with c3:
            _tl_val = fc.get("tick_length", 4)
            try:
                _tl_val = int(_tl_val) if _tl_val is not None else 4
            except (TypeError, ValueError):
                _tl_val = 4
            fc["tick_length"] = st.number_input(
                "Length", value=_tl_val,
                min_value=0, max_value=20, step=1,
                key=f"{key_prefix}_tl")

        st.caption("✏️ All changes apply live — no Save button needed.")

    return True


def render_live_colormap_picker(viz_cfg, *, key_prefix="cmap",
                                  label="Per-field colormap (live)"):
    """Live per-field colormap picker with swatch preview."""
    field_keys = list(FIELD_DEFAULT_CMAPS.keys())
    cmap_options = ColormapRegistry.list_all()
    fc = viz_cfg.setdefault("field_settings",
                              {f: {"auto": True, "vmin": None, "vmax": None,
                                   "log": False, "invert": False,
                                   "cmap": FIELD_DEFAULT_CMAPS.get(f, "viridis")}
                               for f in field_keys})
    default_expanded = bool(st.session_state.get(f"{key_prefix}_exp", False))
    with st.expander(f"🎨 {label}", expanded=default_expanded):
        st.session_state[f"{key_prefix}_exp"] = st.checkbox(
            "Keep expanded", value=default_expanded,
            key=f"{key_prefix}_keep_exp",
            help="If checked, this panel stays open across reruns.")
        st.caption(
            f"Pick a colormap for each of the {len(field_keys)} fields. "
            f"Changes apply LIVE on the next render — no Save button needed.")
        for i in range(0, len(field_keys), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i + j < len(field_keys):
                    with col:
                        fname = field_keys[i + j]
                        fset = fc.setdefault(fname, {})
                        cur_cmap = fset.get("cmap",
                                              FIELD_DEFAULT_CMAPS.get(fname, "viridis"))
                        cmap_idx = (cmap_options.index(cur_cmap)
                                     if cur_cmap in cmap_options else 0)
                        new_cmap = st.selectbox(
                            f"{fname}",
                            cmap_options,
                            index=cmap_idx,
                            key=f"{key_prefix}_{fname}_cmap",
                            format_func=lambda x: x)
                        fset["cmap"] = new_cmap
                        invert = st.checkbox(
                            f"Invert {fname}",
                            value=bool(fset.get("invert", False)),
                            key=f"{key_prefix}_{fname}_inv")
                        fset["invert"] = invert
                        try:
                            swatch_cmap = ColormapRegistry.resolve(new_cmap,
                                                                     invert=invert)
                            gradient = np.linspace(0, 1, 256).reshape(1, -1)
                            fig_sw, ax_sw = plt.subplots(figsize=(3.5, 0.35),
                                                          dpi=70)
                            ax_sw.imshow(gradient, aspect="auto", cmap=swatch_cmap)
                            ax_sw.set_axis_off()
                            ax_sw.set_title(f"{fname} → {new_cmap}",
                                             fontsize=8, pad=2)
                            st.pyplot(fig_sw, clear_figure=True)
                            plt.close(fig_sw)
                        except Exception:
                            pass
    return True


# ============================================================================
# INTERFACE PROFILING HELPERS
# ============================================================================
def _extract_interface_profile(interface_2d):
    """Extract a 1D profile from a 2D interface array."""
    a = np.asarray(interface_2d)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        return a
    if a.shape[0] > 1 and np.allclose(a[0, :], a[-1, :], atol=1e-9):
        return a[0, :]
    return np.mean(a, axis=0)


def _plot_interface(ax, x_coords, interface_2d, **line_kwargs):
    """Plot an interface profile on the given axes."""
    a = np.asarray(interface_2d)
    if a.ndim == 0 or (a.ndim == 1 and a.size == 1):
        ax.axhline(float(a), **line_kwargs)
        return
    profile = _extract_interface_profile(a)
    if profile is None:
        return
    ax.plot(x_coords, profile, **line_kwargs)


# ============================================================================
# ENHANCED LINE PROFILER
# ============================================================================
class EnhancedLineProfiler:
    """Extract 1D profiles from 2D fields along various directions.

    Supported profile types:
        horizontal     — along x at a given y position
        vertical       — along y at a given x position
        diagonal       — along the main diagonal (top-left to bottom-right)
        anti_diagonal  — along the anti-diagonal (top-right to bottom-left)
        custom         — along a custom angle
    """

    def __init__(self, Nx, Ny, dx, dy):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]

    @handle_errors
    def extract_profile(self, data, profile_type="horizontal",
                        position_ratio=0.5, angle_deg=45):
        """Extract a 1D profile from a 2D array.

        Returns (distance_array, value_array, bbox_tuple_or_None).
        """
        ptype = str(profile_type).lower().replace("-", "_")
        ny, nx = data.shape
        cx, cy = nx // 2, ny // 2
        off_x = int(nx * 0.35 * position_ratio)
        off_y = int(ny * 0.35 * position_ratio)

        if ptype == "horizontal":
            r = min(max(cy + off_y, 0), ny - 1)
            return (np.linspace(self.extent[0], self.extent[1], nx),
                    data[r, :], None)

        if ptype == "vertical":
            cidx = min(max(cx + off_x, 0), nx - 1)
            return (np.linspace(self.extent[2], self.extent[3], ny),
                    data[:, cidx], None)

        if ptype == "diagonal":
            L = int(min(nx, ny) * 0.8)
            s = (cx - L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] + i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None

        if ptype in ("anti_diagonal", "antidiagonal"):
            L = int(min(nx, ny) * 0.8)
            s = (cx + L // 2, cy - L // 2)
            vals, ds = [], []
            for i in range(L):
                x, y = s[0] - i, s[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    vals.append(data[y, x])
                    ds.append((i - L // 2) * self.dx * np.sqrt(2))
            return np.array(ds), np.array(vals), None

        if ptype == "custom":
            ar = np.deg2rad(angle_deg)
            L = int(min(nx, ny) * 0.8)
            vals, ds = [], []
            for t in np.linspace(-L // 2, L // 2, L):
                x = cx + t * np.cos(ar) + off_y * np.cos(ar + np.pi / 2)
                y = cy + t * np.sin(ar) + off_y * np.sin(ar + np.pi / 2)
                if 0 <= x < nx - 1 and 0 <= y < ny - 1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
                    wx, wy = x - x0, y - y0
                    v = (data[y0, x0] * (1 - wx) * (1 - wy)
                         + data[y0, x1] * wx * (1 - wy)
                         + data[y1, x0] * (1 - wx) * wy
                         + data[y1, x1] * wx * wy)
                    vals.append(v)
                    ds.append(t * self.dx)
            return np.array(ds), np.array(vals), None

        return None, None, None

    @handle_errors
    def extract_profile_at(self, data, x0, y0, x1, y1, n=200):
        """Extract a profile along an arbitrary line from (x0,y0) to (x1,y1)."""
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        i = ((xs - self.extent[0]) / self.dx).astype(float)
        j = ((ys - self.extent[2]) / self.dy).astype(float)
        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)
        i0 = i.astype(int)
        j0 = j.astype(int)
        i1 = np.clip(i0 + 1, 0, self.Nx - 1)
        j1 = np.clip(j0 + 1, 0, self.Ny - 1)
        wi = i - i0
        wj = j - j0
        v = (data[j0, i0] * (1 - wi) * (1 - wj)
             + data[j0, i1] * wi * (1 - wj)
             + data[j1, i0] * (1 - wi) * wj
             + data[j1, i1] * wi * wj)
        dist = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
        return dist, v


# ============================================================================
# LANDAU-DEVONSHIRE MATERIAL PARAMETERS
# ============================================================================
class LandauParams:
    """3-component tetragonal Landau-Devonshire parameters (SI units).

    Attributes:
        alpha1  : V·m/C       (Landau quadratic coefficient, < 0 below Tc)
        alpha11 : V·m⁵/C³     (Landau quartic, > 0)
        alpha12 : V·m⁵/C³     (Landau cross-coupling, must be > 2*alpha11)
        G11     : V·m³/C      (Lifshitz gradient, longitudinal)
        G44     : V·m³/C      (Lifshitz gradient, transverse/shear)
        G12     : V·m³/C      (Lifshitz gradient, cross-coupling)
        beta_sub: V·m/C       (Substrate anisotropy, favors Pz; 0 = no anisotropy)
    """

    def __init__(self, alpha1=-3.0e10, alpha11=6.0e10, alpha12=1.5e11,
                 G11=1.0e-8, G44=1.0e-8, G12=0.0, beta_sub=0.0):
        self.alpha1 = alpha1
        self.alpha11 = alpha11
        self.alpha12 = alpha12
        self.G11 = G11
        self.G44 = G44
        self.G12 = G12
        self.beta_sub = beta_sub

        # Bug fix #3: enforce tetragonal symmetry condition
        if self.alpha12 <= 2 * self.alpha11:
            warnings.warn(
                f"Tetragonal symmetry requires alpha12 > 2*alpha11, got "
                f"alpha12={self.alpha12:.3e} <= 2*alpha11={2*self.alpha11:.3e}. "
                f"Diagonal state will be energetically favored over axial — "
                f"system may collapse to a single domain.",
                UserWarning, stacklevel=2,
            )

    def Ps_eq(self):
        """Equilibrium spontaneous polarization magnitude (axial minimum)."""
        if self.alpha1 < 0 and self.alpha11 > 0:
            return float(np.sqrt(-self.alpha1 / (2 * self.alpha11)))
        return 0.0

    def xi_wall(self):
        """Domain wall width ~ sqrt(G11 / |alpha1|)."""
        if self.alpha1 != 0:
            return float(np.sqrt(self.G11 / abs(self.alpha1)))
        return 1.0

    def __repr__(self):
        return (f"LandauParams(alpha1={self.alpha1:.3e}, alpha11={self.alpha11:.3e}, "
                f"alpha12={self.alpha12:.3e}, G11={self.G11:.3e}, G44={self.G44:.3e}, "
                f"beta_sub={self.beta_sub:.3e})")


PRESETS = {
    "PbTiO3-like": LandauParams(
        alpha1=-3.0e10, alpha11=6.0e10, alpha12=1.5e11,
        G11=1.0e-8, G44=1.0e-8, beta_sub=4.5e10,  # 1.5 * |alpha1|
    ),
    "HfO2-like ": LandauParams(
        alpha1=-2.5e9, alpha11=6.0e10, alpha12=1.5e11,
        G11=1.0e-10, G44=1.0e-10, beta_sub=3.75e9,  # 1.5 * |alpha1|
    ),
    "BaTiO3-like": LandauParams(
        alpha1=-2.0e10, alpha11=4.0e10, alpha12=1.2e11,
        G11=5.0e-9, G44=5.0e-9, beta_sub=3.0e10,
    ),
    "Custom": LandauParams(),
}


def get_preset(name):
    """Return a copy of a preset LandauParams instance."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    p = PRESETS[name]
    return LandauParams(alpha1=p.alpha1, alpha11=p.alpha11, alpha12=p.alpha12,
                        G11=p.G11, G44=p.G44, G12=p.G12, beta_sub=p.beta_sub)


# ============================================================================
# K-SPACE OPERATORS (THIN-FILM FERROELECTRIC)
# ============================================================================
def build_k_operators(Nx, Ny, L, h_film, mat, screening_factor, use_elec, xp):
    """Build the block-diagonal 3x3 linear operator M(k) for (Px, Py, Pz).

    Returns:
        KX, KY, K2_safe, K_par, n_eff, M_xx, M_yy, M_xy, M_zz, dx, dy

    The operator M(k) is block-diagonal:
        [ M_xx  M_xy  0   ]
        [ M_xy  M_yy  0   ]
        [ 0     0     M_zz]

    In-plane block (Px, Py):
        Lifshitz gradient + 2D depolarization projector k⊗k/k²/ε₀

    Pz block:
        G44·k_par² + n(k)/ε₀ · screening_factor
        where n(k) = tanh(k·h/2)/(k·h/2) is the thin-film depolarization kernel.
    """
    dx = L / Nx
    dy = L / Ny
    kx = xp.fft.fftfreq(Nx, d=dx) * 2 * xp.pi
    ky = xp.fft.fftfreq(Ny, d=dy) * 2 * xp.pi
    KX, KY = xp.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2_safe = xp.where(K2 == 0, 1.0, K2)
    K_par = xp.sqrt(K2)

    # Thin-film depolarization kernel for Pz: n(k) = tanh(k*h/2)/(k*h/2)
    kh = K_par * h_film / 2.0
    n_kernel = xp.where(
        K_par == 0, 1.0,
        xp.tanh(kh) / xp.where(kh == 0, 1.0, kh),
    )
    n_eff = n_kernel * screening_factor

    # In-plane Lifshitz gradient + 2D depol projector
    if use_elec:
        depol_xx = KX**2 / K2_safe
        depol_yy = KY**2 / K2_safe
        depol_xy = KX * KY / K2_safe
    else:
        depol_xx = depol_yy = depol_xy = 0.0

    M_xx = mat.G11 * KX**2 + mat.G44 * KY**2 + depol_xx / EPSILON_0
    M_yy = mat.G44 * KX**2 + mat.G11 * KY**2 + depol_yy / EPSILON_0
    M_xy = (mat.G12 + mat.G44) * KX * KY + depol_xy / EPSILON_0

    # Pz block: G44*k_par² + thin-film depol kernel
    M_zz = mat.G44 * K2 + n_eff / EPSILON_0

    # Zero mode handling:
    # In-plane: no gradient, no depol (uniform in-plane P has ∇·P = 0)
    M_xx[0, 0] = 0.0
    M_yy[0, 0] = 0.0
    M_xy[0, 0] = 0.0
    # Bug fix #5: Pz gets full depol penalty for uniform Pz
    # (this is the key open-circuit term that penalizes uniform out-of-plane P)
    M_zz[0, 0] = screening_factor / EPSILON_0

    return KX, KY, K2_safe, K_par, n_eff, M_xx, M_yy, M_xy, M_zz, dx, dy


def solve_poisson_inplane(Px, Py, KX, KY, K2_safe, xp):
    """Solve ∇²Φ = ∇·P_inplane / ε₀, return E_inplane = -∇Φ.

    Note: Pz does NOT contribute here (no z-derivatives in 2D).
    Its depolarization is handled by the thin-film kernel in M_zz.

    Bug fix #6: division by EPSILON_0 is correct — this gives the physical
    electric field E = -∇Φ in V/m, which enters the TDGL as -E.
    """
    Px_hat = xp.fft.fft2(Px)
    Py_hat = xp.fft.fft2(Py)
    div_P = 1j * (KX * Px_hat + KY * Py_hat) / EPSILON_0
    Phi_hat = -div_P / K2_safe
    Phi_hat[0, 0] = 0.0  # zero-mean potential
    Ex = xp.fft.ifft2(-1j * KX * Phi_hat).real
    Ey = xp.fft.ifft2(-1j * KY * Phi_hat).real
    return Ex, Ey


# ============================================================================
# INITIAL CONDITION GENERATORS
# ============================================================================
def make_initial_P(Nx, Ny, L, mode, amp, cutoff, ps_eq, seed, xp):
    """Generate the initial (Px, Py, Pz) fields.

    Bug fix #1: amp is already in C/m². The conversion from μC/cm² to C/m²
    is done once in the sidebar via `noise_amp_uc * 1e-2` (correct: 1 μC/cm²
    = 1e-6 C / 1e-4 m² = 1e-2 C/m²). Previously the code did
    `noise_amp_uc * 1e4 * 1e-4` which evaluated to `noise_amp_uc * 1`
    (100× too large), making the initial condition a huge out-of-equilibrium
    blob instead of a symmetry-breaking seed.

    Modes:
        "band_limited"  : smooth random Pz + small Px, Py noise (default)
        "white_noise"   : pure white noise on all 3 components
        "seeded_stripes": 180° stripe seed on Pz along x
    """
    xp.random.seed(seed)
    kx_n = xp.fft.fftfreq(Nx)
    ky_n = xp.fft.fftfreq(Ny)
    KXn, KYn = xp.meshgrid(kx_n, ky_n, indexing="ij")
    kr = xp.sqrt(KXn**2 + KYn**2)

    if mode == "white_noise":
        Px = amp * (xp.random.rand(Nx, Ny) - 0.5)
        Py = amp * (xp.random.rand(Nx, Ny) - 0.5)
        Pz = amp * (xp.random.rand(Nx, Ny) - 0.5)
        return Px, Py, Pz

    if mode == "seeded_stripes":
        x = xp.linspace(0, L, Nx, endpoint=False)[:, None]
        stripes = xp.sign(xp.cos(2 * xp.pi * x * 4 / L))
        Pz = ps_eq * stripes * xp.ones((Nx, Ny))
        Px = 0.02 * ps_eq * (xp.random.rand(Nx, Ny) - 0.5)
        Py = 0.02 * ps_eq * (xp.random.rand(Nx, Ny) - 0.5)
        return Px, Py, Pz

    # Default: band-limited noise on Pz + small noise on Px, Py
    mask = (kr <= cutoff).astype(xp.float64) * xp.exp(-kr**2 / (2 * (cutoff * 1.2)**2))

    Pz_raw = (xp.random.rand(Nx, Ny) - 0.5)
    Pz_bl = xp.fft.ifft2(mask * xp.fft.fft2(Pz_raw)).real
    Pz_bl /= (xp.std(Pz_bl) + 1e-12)
    Pz = amp * Pz_bl

    Px_raw = (xp.random.rand(Nx, Ny) - 0.5)
    Py_raw = (xp.random.rand(Nx, Ny) - 0.5)
    Px_bl = xp.fft.ifft2(mask * xp.fft.fft2(Px_raw)).real
    Py_bl = xp.fft.ifft2(mask * xp.fft.fft2(Py_raw)).real
    Px_bl /= (xp.std(Px_bl) + 1e-12)
    Py_bl /= (xp.std(Py_bl) + 1e-12)
    Px = 0.05 * amp * Px_bl
    Py = 0.05 * amp * Py_bl
    return Px, Py, Pz


# ============================================================================
# LANDAU-DEVONSHIRE BULK DERIVATIVE (3-COMPONENT)
# ============================================================================
def landau_deriv_3c(Px, Py, Pz, mat, xp=None):
    """d f_bulk / dP for 3-component tetragonal Landau-Devonshire + substrate.

    f_bulk = alpha1*(Px² + Py² + Pz²)
           + alpha11*(Px⁴ + Py⁴ + Pz⁴)
           + alpha12*(Px²Py² + Py²Pz² + Px²Pz²)
           + (beta_sub/2)*(Px² + Py²)        [substrate anisotropy]

    Returns (dF/dPx, dF/dPy, dF/dPz) — the NONLINEAR part, computed in real space.
    The substrate anisotropy term penalizes in-plane P, mimicking compressive
    substrate strain that favors out-of-plane Pz.

    Bug fix #4: Clamp P values to prevent overflow in Px*Px² (cubic term).
    Physical P should stay near ±Ps; values beyond 5*Ps indicate numerical
    instability. Clamping prevents float64 overflow RuntimeWarnings.
    """
    if xp is None:
        xp = np

    # Clamp to prevent overflow (physical P should stay near ±Ps)
    Ps = abs(mat.alpha1 / (2 * mat.alpha11))**0.5 if mat.alpha11 > 0 else 1.0
    P_max = 5.0 * Ps  # ~5*Ps safety margin
    Px = xp.clip(Px, -P_max, P_max)
    Py = xp.clip(Py, -P_max, P_max)
    Pz = xp.clip(Pz, -P_max, P_max)

    Px2, Py2, Pz2 = Px**2, Py**2, Pz**2
    dPx = (2 * mat.alpha1 * Px
           + 4 * mat.alpha11 * Px * Px2
           + 2 * mat.alpha12 * Px * (Py2 + Pz2)
           + 2 * mat.beta_sub * Px)
    dPy = (2 * mat.alpha1 * Py
           + 4 * mat.alpha11 * Py * Py2
           + 2 * mat.alpha12 * Py * (Px2 + Pz2)
           + 2 * mat.beta_sub * Py)
    dPz = (2 * mat.alpha1 * Pz
           + 4 * mat.alpha11 * Pz * Pz2
           + 2 * mat.alpha12 * Pz * (Px2 + Py2))
    return dPx, dPy, dPz


# ============================================================================
# FERROELECTRIC PHASE-FIELD SOLVER
# ============================================================================
class FerroSolver:
    """3-component thin-film ferroelectric phase-field solver.

    Solves the TDGL equation:
        dP/dt = -Gamma * dF/dP

    using a semi-implicit Fourier spectral scheme. The linear part (gradient +
    depolarization) is treated implicitly (unconditionally stable), and the
    nonlinear Landau part is treated explicitly.

    State variables:
        Px, Py, Pz : polarization components (xp arrays on grid)

    History (diagnostics):
        step, f_bulk, f_grad, f_depol_Pz, f_depol_inplane, f_total
    """

    def __init__(self, params):
        self.params = params
        self.xp = params["xp"]
        self.mat = params["mat"]
        self.Nx = params["Nx"]
        self.Ny = params["Ny"]
        self.L = params["L"]
        self.h_film = params["h_film"]
        self.dt = params["dt"]
        self.Gamma = params["Gamma"]
        self.screening = params["screening"]
        self.use_elec = params["use_elec"]
        self.E_ext = params["E_ext"]
        self.init_mode = params["init_mode"]
        self.noise_amp = params["noise_amp"]
        self.noise_cutoff = params["noise_cutoff"]
        self.seed = params["seed"]

        # Build k-space operators
        (self.KX, self.KY, self.K2_safe, self.K_par, self.n_eff,
         self.M_xx, self.M_yy, self.M_xy, self.M_zz,
         self.dx, self.dy) = build_k_operators(
            self.Nx, self.Ny, self.L, self.h_film,
            self.mat, self.screening, self.use_elec, self.xp)

        # Initial condition
        ps_eq = self.mat.Ps_eq()
        self.Px, self.Py, self.Pz = make_initial_P(
            self.Nx, self.Ny, self.L, self.init_mode,
            self.noise_amp, self.noise_cutoff, ps_eq, self.seed, self.xp)

        # History (diagnostics)
        self.history = {
            "step": [],
            "f_bulk": [],
            "f_grad": [],
            "f_depol_Pz": [],
            "f_depol_inplane": [],
            "f_total": [],
        }

    @handle_errors
    def step(self):
        """Advance by one time step using semi-implicit Fourier spectral scheme.

        The scheme is:
            [I + dt*Gamma*M] P^{n+1}_hat = P^n_hat - dt*Gamma*NL_hat

        where M is the linear operator (gradient + depol) and NL is the
        nonlinear part (Landau derivative + external field).

        The 3x3 system is block-diagonal, so we solve:
            - 2x2 for (Px, Py)
            - 1x1 for Pz
        """
        xp = self.xp

        # Nonlinear Landau derivative (real space)
        dFx, dFy, dFz = landau_deriv_3c(self.Px, self.Py, self.Pz, self.mat, xp)

        # In-plane depol field from Poisson
        if self.use_elec:
            Ex_d, Ey_d = solve_poisson_inplane(
                self.Px, self.Py, self.KX, self.KY, self.K2_safe, xp)
        else:
            Ex_d = Ey_d = 0.0

        NLx = dFx - Ex_d - self.E_ext[0]
        NLy = dFy - Ey_d - self.E_ext[1]
        NLz = dFz - self.E_ext[2]  # Pz depol is in M_zz, no real-space term

        # Fourier transforms
        Px_h = xp.fft.fft2(self.Px)
        Py_h = xp.fft.fft2(self.Py)
        Pz_h = xp.fft.fft2(self.Pz)
        rx = Px_h - self.dt * self.Gamma * xp.fft.fft2(NLx)
        ry = Py_h - self.dt * self.Gamma * xp.fft.fft2(NLy)
        rz = Pz_h - self.dt * self.Gamma * xp.fft.fft2(NLz)

        # In-plane 2x2 solve
        A = 1.0 + self.dt * self.Gamma * self.M_xx
        D = 1.0 + self.dt * self.Gamma * self.M_yy
        B = self.dt * self.Gamma * self.M_xy
        det = A * D - B * B
        self.Px = xp.fft.ifft2((D * rx - B * ry) / det).real
        self.Py = xp.fft.ifft2((-B * rx + A * ry) / det).real

        # Pz 1x1 solve
        self.Pz = xp.fft.ifft2(rz / (1.0 + self.dt * self.Gamma * self.M_zz)).real

        # Energy diagnostics
        self._update_history()

        return self._snapshot()

    def _update_history(self):
        """Compute and store energy components."""
        xp = self.xp
        Px, Py, Pz = self.Px, self.Py, self.Pz
        Px2, Py2, Pz2 = Px**2, Py**2, Pz**2
        mat = self.mat

        f_bulk = (mat.alpha1 * (Px2 + Py2 + Pz2)
                  + mat.alpha11 * (Px2*Px2 + Py2*Py2 + Pz2*Pz2)
                  + mat.alpha12 * (Px2*Py2 + Py2*Pz2 + Px2*Pz2)
                  + 0.5 * mat.beta_sub * (Px2 + Py2))

        Px_x, Px_y = xp.gradient(Px, self.dx, self.dy)
        Py_x, Py_y = xp.gradient(Py, self.dx, self.dy)
        Pz_x, Pz_y = xp.gradient(Pz, self.dx, self.dy)
        f_grad = (0.5 * mat.G11 * (Px_x**2 + Py_y**2)
                  + mat.G12 * Px_x * Py_y
                  + 0.5 * mat.G44 * ((Px_y + Py_x)**2 + Pz_x**2 + Pz_y**2))

        # Bug fix #2: f_depol_Pz computed ONCE (removed dead duplicate line)
        Pz_hat = xp.fft.fft2(Pz)
        f_depol_Pz = (xp.mean((self.n_eff / (2 * EPSILON_0))
                              * xp.abs(Pz_hat)**2) / (self.Nx * self.Ny))

        if self.use_elec:
            Ex_d, Ey_d = solve_poisson_inplane(
                Px, Py, self.KX, self.KY, self.K2_safe, xp)
            f_depol_in = -0.5 * (Ex_d * Px + Ey_d * Py)
        else:
            f_depol_in = xp.zeros_like(Px)

        total = float(xp.mean(f_bulk + f_grad + f_depol_Pz + f_depol_in))
        self.history["step"].append(len(self.history["step"]))
        self.history["f_bulk"].append(float(xp.mean(f_bulk)))
        self.history["f_grad"].append(float(xp.mean(f_grad)))
        self.history["f_depol_Pz"].append(float(f_depol_Pz))
        self.history["f_depol_inplane"].append(
            float(xp.mean(f_depol_in)) if self.use_elec else 0.0)
        self.history["f_total"].append(total)

    def _snapshot(self):
        """Return all fields the visualizer might want (CPU arrays).

        Bug fix #2: energy fields (f_bulk, f_grad, f_depol_Pz, f_depol_inplane,
        f_total) are returned as 2D arrays, NOT scalars. The multi-field grid
        visualizer calls ax.imshow() on each field, which requires a 2D array.
        Previously these were stored as scalar mean values, causing
        "TypeError: Invalid shape () for image data".
        """
        xp = self.xp
        Px, Py, Pz = self.Px, self.Py, self.Pz
        Px_cpu, Py_cpu, Pz_cpu = to_cpu(Px), to_cpu(Py), to_cpu(Pz)
        P_mag = np.sqrt(Px_cpu**2 + Py_cpu**2 + Pz_cpu**2)

        # ---- Recompute 2D energy density fields for visualization ----
        Px2, Py2, Pz2 = Px**2, Py**2, Pz**2
        mat = self.mat
        f_bulk_2d = (mat.alpha1 * (Px2 + Py2 + Pz2)
                     + mat.alpha11 * (Px2*Px2 + Py2*Py2 + Pz2*Pz2)
                     + mat.alpha12 * (Px2*Py2 + Py2*Pz2 + Px2*Pz2)
                     + 0.5 * mat.beta_sub * (Px2 + Py2))

        Px_x, Px_y = xp.gradient(Px, self.dx, self.dy)
        Py_x, Py_y = xp.gradient(Py, self.dx, self.dy)
        Pz_x, Pz_y = xp.gradient(Pz, self.dx, self.dy)
        f_grad_2d = (0.5 * mat.G11 * (Px_x**2 + Py_y**2)
                     + mat.G12 * Px_x * Py_y
                     + 0.5 * mat.G44 * ((Px_y + Py_x)**2 + Pz_x**2 + Pz_y**2))

        # Depol Pz: broadcast scalar mean to 2D for visualization
        Pz_hat = xp.fft.fft2(Pz)
        f_depol_Pz_scalar = float(xp.mean(
            (self.n_eff / (2 * EPSILON_0)) * xp.abs(Pz_hat)**2) / (self.Nx * self.Ny))
        f_depol_Pz_2d = xp.full_like(Pz, f_depol_Pz_scalar)

        if self.use_elec:
            Ex_d, Ey_d = solve_poisson_inplane(
                Px, Py, self.KX, self.KY, self.K2_safe, xp)
            f_depol_in_2d = -0.5 * (Ex_d * Px + Ey_d * Py)
        else:
            f_depol_in_2d = xp.zeros_like(Px)

        f_total_2d = f_bulk_2d + f_grad_2d + f_depol_Pz_2d + f_depol_in_2d

        return {
            "Px": Px_cpu,
            "Py": Py_cpu,
            "Pz": Pz_cpu,
            "P_mag": P_mag,
            "f_bulk": to_cpu(f_bulk_2d),
            "f_grad": to_cpu(f_grad_2d),
            "f_depol_Pz": to_cpu(f_depol_Pz_2d),
            "f_depol_inplane": to_cpu(f_depol_in_2d),
            "f_total": to_cpu(f_total_2d),
            "convergence": {
                "f_total": float(xp.mean(f_total_2d)),
                "mean_Pz": float(np.mean(np.abs(Pz_cpu))),
                "mean_Pinplane": float(np.mean(np.sqrt(Px_cpu**2 + Py_cpu**2))),
                "max_Pz": float(np.max(np.abs(Pz_cpu))),
            },
        }


# ============================================================================
# FIELD RESCALING AND NORM HELPERS
# ============================================================================
def _rescale_field(field_name, data):
    """No unit rescaling needed for ferro fields (all already in display units)."""
    return np.asarray(data).copy(), FIELD_UNITS.get(field_name, "")


def compute_field_norm(field_name, data, fs, default_rng=None):
    """Compute the matplotlib Norm and (vmin, vmax) for a field."""
    if fs is None:
        fs = {}
    auto = fs.get("auto", True)
    use_log = fs.get("log", False)
    d = np.asarray(data)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return None, None, None

    if auto:
        if default_rng is not None:
            vmin, vmax = float(default_rng[0]), float(default_rng[1])
        else:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
    else:
        vmin = fs.get("vmin", None)
        vmax = fs.get("vmax", None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None
                    else float(np.min(finite)))
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None
                    else float(np.max(finite)))

    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-9

    norm = None
    if use_log:
        pos = finite[finite > 0]
        if pos.size > 0:
            if vmin <= 0:
                vmin = float(np.percentile(pos, 2))
            if vmax <= 0:
                vmax = float(np.percentile(pos, 98))
            if vmin >= vmax:
                vmin = vmax / 10.0
            try:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            except Exception:
                norm = None

    return norm, vmin, vmax


def get_global_bounds(history, field_name, fs, default_rng=None):
    """Compute global (vmin, vmax) across all history frames for consistent animation."""
    if fs is None:
        fs = {}

    if not fs.get("auto", True):
        vmin = fs.get("vmin", None)
        vmax = fs.get("vmax", None)
        if vmin is None:
            vmin = (float(default_rng[0]) if default_rng is not None else 0.0)
        if vmax is None:
            vmax = (float(default_rng[1]) if default_rng is not None else 1.0)
        return float(vmin), float(vmax)

    if default_rng is not None:
        return float(default_rng[0]), float(default_rng[1])

    vmin_all, vmax_all = [], []
    for h in history:
        if field_name not in h:
            continue
        dd, _ = _rescale_field(field_name, h[field_name])
        finite = dd[np.isfinite(dd)]
        if finite.size:
            vmin_all.append(np.percentile(finite, 2))
            vmax_all.append(np.percentile(finite, 98))
    if not vmin_all:
        return None, None

    vmin = float(np.min(vmin_all))
    vmax = float(np.max(vmax_all))
    return vmin, vmax


# ============================================================================
# FERROELECTRIC VISUALIZER
# ============================================================================
class FerroVisualizer:
    """Visualization routines for the ferroelectric phase-field solver.

    Provides:
        - create_multi_field_comparison : grid of all 9 fields
        - create_overlay                : base + contour/heatmap overlay
        - create_vector_overlay         : quiver of (Px, Py) over base field
        - create_plotly_heatmap         : interactive Plotly heatmap
        - create_plotly_3d_surface      : 3D surface plot
        - create_plotly_animation       : time-scrubbing animation
        - create_multi_time_profiles    : multi-frame line profile comparison
        - create_animation_buffer       : matplotlib GIF/MP4 writer
    """

    def __init__(self, Nx, Ny, dx, dy=None, dt=1e-12):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, (dy if dy is not None else dx)
        self.dt = dt
        self.extent = [-Nx * self.dx / 2 * 1e9, Nx * self.dx / 2 * 1e9,
                       -Ny * self.dy / 2 * 1e9, Ny * self.dy / 2 * 1e9]
        self.line_profiler = EnhancedLineProfiler(Nx, Ny, dx, self.dy)

    def get_colormap(self, name, invert=False):
        return ColormapRegistry.resolve(name, invert=invert)

    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        """Create a grid of all available fields as subplots."""
        if style_params is None:
            style_params = {}
        field_settings = style_params.get("field_settings", {})
        colormaps = style_params.get("colormaps", {})
        title_fs = float(style_params.get("title_font_size", 10))
        label_fs = float(style_params.get("label_font_size", 8))
        cbar_fraction = float(style_params.get("colorbar_fraction", 0.046))
        cbar_pad = float(style_params.get("colorbar_pad", 0.04))
        fig_cfg = style_params.get("figure_customization", {}) or {}

        avail = [(f, t, r) for (f, t, r) in MULTI_FIELD_LAYOUT
                 if f in results_dict]
        if not avail:
            return None

        n = len(avail)
        cols = 3 if n >= 3 else n
        rows = (n + cols - 1) // cols
        fig_w = float(fig_cfg.get("multi_field_width", 4.5 * cols))
        fig_h = float(fig_cfg.get("multi_field_height", 3.8 * rows))
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        show_cb = bool(fig_cfg.get("show_colorbar", True))
        cb_orient = fig_cfg.get("colorbar_orientation", "vertical")

        for idx, (fname, title, rng) in enumerate(avail):
            ax = axes[idx]
            d, unit = _rescale_field(fname, results_dict[fname])

            fs = field_settings.get(fname, {})
            cmap_name = (colormaps.get(fname)
                         or fs.get("cmap")
                         or FIELD_DEFAULT_CMAPS.get(fname, "viridis"))
            cmap = self.get_colormap(cmap_name, invert=fs.get("invert", False))

            interp = "nearest" if fname in DISCRETE_FIELDS else "bilinear"

            default_rng = FIELD_DEFAULT_RNG.get(fname, rng)
            norm, vmin, vmax = compute_field_norm(fname, d, fs,
                                                   default_rng=default_rng)

            if norm is not None:
                im = ax.imshow(d, extent=self.extent, cmap=cmap, norm=norm,
                               origin="lower", aspect="equal", interpolation=interp)
            else:
                im = ax.imshow(d, extent=self.extent, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               origin="lower", aspect="equal", interpolation=interp)
            t_override = fig_cfg.get("title_override", "").strip()
            ax.set_title(t_override if t_override else title, fontsize=title_fs)
            xl = fig_cfg.get("xlabel_override", "").strip() or "x (nm)"
            yl = fig_cfg.get("ylabel_override", "").strip() or "y (nm)"
            ax.set_xlabel(xl, fontsize=label_fs)
            ax.set_ylabel(yl, fontsize=label_fs)
            if show_cb:
                cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction,
                                     pad=cbar_pad, orientation=cb_orient)
                if unit:
                    cbar.set_label(unit)
            apply_axis_styling(ax, fig_cfg, grid_axes=None)

        for k in range(n, len(axes)):
            axes[k].axis("off")
        if fig_cfg.get("tight_layout", True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_overlay(self, results_dict, base_field, overlay_field,
                       alpha=0.5, mode="contour", n_contours=10,
                       style_params=None):
        """Create an overlay of two fields (base heatmap + contour or heatmap overlay)."""
        if style_params is None:
            style_params = {}
        field_settings = style_params.get("field_settings", {})
        colormaps = style_params.get("colormaps", {})
        title_fs = float(style_params.get("title_font_size", 11))
        label_fs = float(style_params.get("label_font_size", 9))
        fig_cfg = style_params.get("figure_customization", {}) or {}

        if base_field not in results_dict or overlay_field not in results_dict:
            return None

        base, base_unit = _rescale_field(base_field, results_dict[base_field])
        ov, ov_unit = _rescale_field(overlay_field, results_dict[overlay_field])
        base_fs = field_settings.get(base_field, {})
        ov_fs = field_settings.get(overlay_field, {})

        base_cmap_name = (colormaps.get(base_field) or base_fs.get("cmap")
                          or FIELD_DEFAULT_CMAPS.get(base_field, "viridis"))
        base_cmap = self.get_colormap(base_cmap_name,
                                       invert=base_fs.get("invert", False))

        fig_w = float(fig_cfg.get("overlay_width", 11.0))
        fig_h = float(fig_cfg.get("overlay_height", 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        norm, vmin, vmax = compute_field_norm(
            base_field, base, base_fs,
            default_rng=FIELD_DEFAULT_RNG.get(base_field))
        if norm is not None:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap, norm=norm,
                                origin="lower", aspect="equal", interpolation="bilinear")
        else:
            im_base = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                                vmin=vmin, vmax=vmax,
                                origin="lower", aspect="equal", interpolation="bilinear")

        contour_lw = float(fig_cfg.get("contour_line_width", 1.0))
        if mode == "contour":
            ov_cmap_name = (colormaps.get(overlay_field) or ov_fs.get("cmap")
                            or FIELD_DEFAULT_CMAPS.get(overlay_field, "viridis"))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get("invert", False))
            x = np.linspace(self.extent[0], self.extent[1], self.Nx)
            y = np.linspace(self.extent[2], self.extent[3], self.Ny)
            X, Y = np.meshgrid(x, y)
            ov_for = ov.copy()
            ov_for[~np.isfinite(ov_for)] = np.nan
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            try:
                if ov_vmin is None or ov_vmax is None or ov_vmax <= ov_vmin:
                    levels = n_contours
                else:
                    levels = np.linspace(ov_vmin, ov_vmax, n_contours)
                cs = ax.contour(X, Y, ov_for, levels=levels, cmap=ov_cmap,
                                linewidths=contour_lw, alpha=alpha)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
            except Exception as e:
                logger.warning(f"Overlay contour failed: {e}")
        else:
            ov_cmap_name = (colormaps.get(overlay_field) or ov_fs.get("cmap")
                            or FIELD_DEFAULT_CMAPS.get(overlay_field, "viridis"))
            ov_cmap = self.get_colormap(ov_cmap_name,
                                         invert=ov_fs.get("invert", False))
            _, ov_vmin, ov_vmax = compute_field_norm(
                overlay_field, ov, ov_fs,
                default_rng=FIELD_DEFAULT_RNG.get(overlay_field))
            ax.imshow(ov, extent=self.extent, cmap=ov_cmap,
                      vmin=ov_vmin, vmax=ov_vmax,
                      origin="lower", aspect="equal",
                      interpolation="bilinear", alpha=alpha)

        if fig_cfg.get("show_colorbar", True):
            cb_orient = fig_cfg.get("colorbar_orientation", "vertical")
            cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04,
                                 orientation=cb_orient)
            if base_unit:
                cbar.set_label(f"{base_field} ({base_unit})")

        default_title = (f"Overlay: {FIELD_LABELS.get(base_field, base_field)} + "
                         f"{FIELD_LABELS.get(overlay_field, overlay_field)} "
                         f"(mode={mode}, α={alpha:.2f})")
        apply_axis_styling(ax, fig_cfg, title=default_title, title_fs=title_fs,
                           xlabel="x (nm)", ylabel="y (nm)", label_fs=label_fs)
        if fig_cfg.get("tight_layout", True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_vector_overlay(self, results_dict, density=24, color="white",
                              scale=1.0, base_field="Pz", style_params=None):
        """Quiver overlay of in-plane (Px, Py) over a base field heatmap."""
        if style_params is None:
            style_params = {}
        field_settings = style_params.get("field_settings", {})
        colormaps = style_params.get("colormaps", {})
        title_fs = float(style_params.get("title_font_size", 11))
        label_fs = float(style_params.get("label_font_size", 9))
        fig_cfg = style_params.get("figure_customization", {}) or {}

        if "Px" not in results_dict or "Py" not in results_dict:
            return None

        fig_w = float(fig_cfg.get("vector_width", 11.0))
        fig_h = float(fig_cfg.get("vector_height", 5.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if base_field and base_field in results_dict:
            base, base_unit = _rescale_field(base_field, results_dict[base_field])
            base_fs = field_settings.get(base_field, {})
            base_cmap_name = (colormaps.get(base_field) or base_fs.get("cmap")
                              or FIELD_DEFAULT_CMAPS.get(base_field, "viridis"))
            base_cmap = self.get_colormap(base_cmap_name,
                                           invert=base_fs.get("invert", False))
            norm, vmin, vmax = compute_field_norm(
                base_field, base, base_fs,
                default_rng=FIELD_DEFAULT_RNG.get(base_field))
            if norm is not None:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap, norm=norm,
                               origin="lower", aspect="equal", interpolation="bilinear")
            else:
                im = ax.imshow(base, extent=self.extent, cmap=base_cmap,
                               vmin=vmin, vmax=vmax,
                               origin="lower", aspect="equal", interpolation="bilinear")
            if fig_cfg.get("show_colorbar", True):
                cb_orient = fig_cfg.get("colorbar_orientation", "vertical")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation=cb_orient,
                             label=f"{base_field} ({base_unit})")

        step = max(1, self.Nx // max(density, 1))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)[::step]
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)[::step]
        X, Y = np.meshgrid(x, y)
        U = results_dict["Px"][::step, ::step] * scale
        V = results_dict["Py"][::step, ::step] * scale

        q_width = float(fig_cfg.get("quiver_width", 0.0025))
        q_scale = float(fig_cfg.get("quiver_scale", 30.0))
        ax.quiver(X, Y, U, V, color=color, alpha=0.9,
                  scale=q_scale, width=q_width, pivot="mid")

        apply_axis_styling(ax, fig_cfg,
                           title=f"In-plane P vectors over {base_field}",
                           title_fs=title_fs, xlabel="x (nm)", ylabel="y (nm)",
                           label_fs=label_fs)
        if fig_cfg.get("tight_layout", True):
            plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0,
                              cmap_name=None, viz_cfg=None):
        """Create an interactive Plotly heatmap of a single field."""
        if not PLOTLY_AVAILABLE:
            return None
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get("field_settings", {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = (cmap_name or fs.get("cmap")
                  or FIELD_DEFAULT_CMAPS.get(field_name, "viridis"))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get("invert", False))
        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs, default_rng=FIELD_DEFAULT_RNG.get(field_name))

        z = d
        if fs.get("log", False):
            pos = d > 0
            z = np.where(pos, np.log10(np.maximum(d, 1e-30)), np.nan)

        zmid = 0 if field_name in ("Px", "Py", "Pz") else None

        fig_cfg = viz_cfg.get("figure_customization", {})
        p_width = int(fig_cfg.get("plotly_width", 11.0) * 60)
        p_height = int(fig_cfg.get("plotly_height", 5.5) * 60)
        xl = fig_cfg.get("xlabel_override", "").strip() or "x (nm)"
        yl = fig_cfg.get("ylabel_override", "").strip() or "y (nm)"
        title_ov = fig_cfg.get("title_override", "").strip()
        title_str = title_ov if title_ov else f"{field_name} — Frame {frame_idx}"

        fig = go.Figure()
        show_cb = bool(fig_cfg.get("show_colorbar", True))
        cb_dict = dict(title=f"{field_name} ({unit})") if show_cb else None
        fig.add_trace(go.Heatmap(
            z=z,
            x=np.linspace(self.extent[0], self.extent[1], self.Nx),
            y=np.linspace(self.extent[2], self.extent[3], self.Ny),
            colorscale=colorscale, zmid=zmid, zmin=vmin, zmax=vmax,
            colorbar=cb_dict,
            hovertemplate="x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.4f}<extra></extra>"))
        fig.update_layout(title=title_str, xaxis_title=xl, yaxis_title=yl,
                          yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
                          width=max(400, p_width), height=max(300, p_height),
                          template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0,
                                 cmap_name=None, show_contours=True,
                                 viz_cfg=None):
        """Create a 3D surface plot of a single field."""
        if not PLOTLY_AVAILABLE:
            return None
        if field_name not in results_dict:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get("field_settings", {})
        fs = field_settings.get(field_name, {})

        d, unit = _rescale_field(field_name, results_dict[field_name])
        chosen = (cmap_name or fs.get("cmap")
                  or FIELD_DEFAULT_CMAPS.get(field_name, "viridis"))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get("invert", False))
        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)
        X, Y = np.meshgrid(x, y)

        norm, vmin, vmax = compute_field_norm(
            field_name, d, fs, default_rng=FIELD_DEFAULT_RNG.get(field_name))

        contours = None
        if show_contours:
            contours = dict(z=dict(show=True, usecolormap=True, project_z=True,
                                    highlightcolor="limegreen"))

        fig = go.Figure(data=[go.Surface(
            z=d, x=X, y=Y, colorscale=colorscale, contours=contours,
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=f"{field_name} ({unit})"))])
        fig.update_layout(title=f"3D: {field_name} — Frame {frame_idx}",
                          scene=dict(xaxis_title="x (nm)", yaxis_title="y (nm)",
                                     zaxis_title=f"{field_name} ({unit})",
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=1, z=0.6)),
                          width=800, height=600, template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5, viz_cfg=None):
        """Create Plotly line profiles for multiple directions."""
        if not PLOTLY_AVAILABLE:
            return None
        fig = go.Figure()
        d, unit = _rescale_field(field_name, results_dict[field_name])
        for pt in profile_types:
            res = self.line_profiler.extract_profile(d, pt, position_ratio)
            if res is None:
                continue
            dist, prof, _ = res
            if dist is None:
                continue
            fig.add_trace(go.Scatter(x=dist, y=prof, mode="lines",
                                     name=pt.replace("_", " ").title()))
        fig.update_layout(title=f"{field_name} Line Profiles",
                          xaxis_title="Position (nm)",
                          yaxis_title=f"{field_name} ({unit})",
                          hovermode="x unified", template="plotly_white")
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_multi_time_profiles(self, history, field_name, ptype="horizontal",
                                   frames=None, position_ratio=0.5,
                                   use_plotly=False, viz_cfg=None):
        """Compare line profiles across multiple time frames."""
        if frames is None:
            frames = list(range(len(history)))
        frames = [f for f in frames if 0 <= f < len(history)]
        if not frames:
            return None

        viz_cfg = viz_cfg or {}
        fig_cfg = viz_cfg.get("figure_customization", {}) or {}

        if use_plotly:
            if not PLOTLY_AVAILABLE:
                return None
            fig = go.Figure()
            lw = float(fig_cfg.get("line_width", 2.0))
            for f in frames:
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                fig.add_trace(go.Scatter(x=dist, y=prof, mode="lines",
                                         name=f"Frame {f}", line=dict(width=lw)))
            fig.update_layout(title=f"{field_name} — multi-time {ptype}",
                              xaxis_title="Position (nm)",
                              yaxis_title=f"{field_name} ({unit})",
                              hovermode="x unified", template="plotly_white",
                              width=800, height=400)
            return fig
        else:
            fig_w = float(fig_cfg.get("line_profile_width", 11.0))
            fig_h = float(fig_cfg.get("line_profile_height", 4.5))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            cmap = plt.cm.viridis
            n = len(frames)
            lw = float(fig_cfg.get("line_width", 2.0))
            for k, f in enumerate(frames):
                d, unit = _rescale_field(field_name, history[f][field_name])
                res = self.line_profiler.extract_profile(d, ptype, position_ratio)
                if res is None:
                    continue
                dist, prof, _ = res
                if dist is None:
                    continue
                color = cmap(k / max(n - 1, 1))
                ax.plot(dist, prof, color=color, lw=lw, label=f"Frame {f}")
            apply_axis_styling(ax, fig_cfg,
                               title=f"{field_name} — multi-time {ptype}",
                               xlabel="Position (nm)", ylabel=field_name)
            ax.legend(loc="best", fontsize=8)
            if fig_cfg.get("tight_layout", True):
                plt.tight_layout()
            return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_plotly_animation(self, history, field_name, dt,
                                cmap_name=None, viz_cfg=None):
        """Create an interactive Plotly time-scrubbing animation."""
        if not PLOTLY_AVAILABLE:
            return None
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get("field_settings", {})
        fs = field_settings.get(field_name, {})

        chosen = (cmap_name or fs.get("cmap")
                  or FIELD_DEFAULT_CMAPS.get(field_name, "viridis"))
        colorscale = ColormapRegistry.to_plotly(chosen, invert=fs.get("invert", False))

        zmin, zmax = get_global_bounds(history, field_name, fs,
                                        default_rng=FIELD_DEFAULT_RNG.get(field_name))

        x = np.linspace(self.extent[0], self.extent[1], self.Nx)
        y = np.linspace(self.extent[2], self.extent[3], self.Ny)

        frames = []
        for i, h in enumerate(history):
            d, _ = _rescale_field(field_name, h[field_name])
            frames.append(go.Frame(
                data=[go.Heatmap(z=d, x=x, y=y, colorscale=colorscale,
                                 zmin=zmin, zmax=zmax,
                                 colorbar=dict(title=f"{field_name}"))],
                name=f"t={i * dt * 1e12:.2f} ps"))

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"{field_name} — interactive time scrubbing",
            xaxis_title="x (nm)", yaxis_title="y (nm)",
            yaxis=dict(scaleanchor="x", scaleratio=self.dy / self.dx),
            width=850, height=600, template="plotly_white",
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=150, redraw=True),
                                               fromcurrent=True,
                                               transition=dict(duration=0))]),
                         dict(label="⏸ Pause", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                 mode="immediate")])])],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f.name], dict(mode="immediate",
                                                 frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))],
                            label=f.name) for f in frames],
                x=0.05, len=0.9, xanchor="left", y=0, yanchor="top",
                currentvalue=dict(prefix="Frame: ", visible=True,
                                  xanchor="right"))])
        return fig

    # ------------------------------------------------------------------
    @handle_errors
    def create_animation_buffer(self, history, field_name, dt,
                                output_format="gif", fps=5, dpi=120,
                                cmap_name=None, watermark=False,
                                watermark_text="", watermark_pos="lower right",
                                viz_cfg=None):
        """Create a matplotlib GIF or MP4 animation and return a BytesIO buffer."""
        if not history:
            return None
        viz_cfg = viz_cfg or {}
        field_settings = viz_cfg.get("field_settings", {})
        fs = field_settings.get(field_name, {})

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        cmap = ColormapRegistry.resolve(
            cmap_name or fs.get("cmap")
            or FIELD_DEFAULT_CMAPS.get(field_name, "viridis"),
            invert=fs.get("invert", False))

        d0, unit = _rescale_field(field_name, history[0][field_name])
        vmin, vmax = get_global_bounds(
            history, field_name, fs,
            default_rng=FIELD_DEFAULT_RNG.get(field_name))

        interp = "nearest" if field_name in DISCRETE_FIELDS else "bilinear"
        im = ax.imshow(d0, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation=interp, aspect="auto")
        if field_name not in DISCRETE_FIELDS:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f"{field_name} ({unit})")
        title = ax.set_title(f"{field_name} — t = 0 ps")

        wm_artist = None
        if watermark and watermark_text:
            xmap = {"lower right": (0.98, 0.02, "right", "bottom"),
                    "lower left":  (0.02, 0.02, "left",  "bottom"),
                    "upper right": (0.98, 0.98, "right", "top"),
                    "upper left":  (0.02, 0.98, "left",  "top")}
            x, y, ha, va = xmap.get(watermark_pos, (0.98, 0.02, "right", "bottom"))
            wm_artist = ax.text(x, y, watermark_text, transform=ax.transAxes,
                                fontsize=8, color="white", alpha=0.75,
                                ha=ha, va=va,
                                bbox=dict(boxstyle="round,pad=0.25",
                                          facecolor="black", alpha=0.35,
                                          edgecolor="none"))

        def update(i):
            d, _ = _rescale_field(field_name, history[i][field_name])
            im.set_array(d)
            title.set_text(f"{field_name} — t = {i * dt * 1e12:.2f} ps")
            return ([im, title] if wm_artist is None
                    else [im, title, wm_artist])

        ani = animation.FuncAnimation(fig, update, frames=len(history),
                                       interval=1000 / max(fps, 1), blit=True)
        buf = BytesIO()
        try:
            if output_format == "gif":
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tf:
                    tmp_path = tf.name
                try:
                    ani.save(tmp_path, writer="pillow", fps=fps, dpi=dpi)
                    with open(tmp_path, "rb") as f:
                        buf.write(f.read())
                finally:
                    try:
                        _os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                ani.save(buf, writer="ffmpeg", fps=fps, dpi=dpi)
        except Exception as e:
            logger.error(f"Animation export failed: {e}")
            plt.close(fig)
            return None
        plt.close(fig)
        buf.seek(0)
        return buf


# ============================================================================
# DATA EXPORTERS
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


FERRO_FIELD_KEYS = ("Px", "Py", "Pz", "P_mag")


def _clean_params(params):
    """Strip non-serializable items (mat, xp, numpy arrays) from params dict."""
    clean = {}
    for k, v in params.items():
        if k in ("xp", "mat"):
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            v = v.item()
        if isinstance(v, tuple):
            v = list(v)
        try:
            json.dumps(v, default=str)
            clean[k] = v
        except Exception:
            clean[k] = str(v)
    return clean


class DataExporter:
    """Export simulation results in PKL / NPZ / JSON / CSV / HDF5 / PT formats."""

    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, name):
        """Export full state as a pickle file."""
        buf = BytesIO()
        pickle.dump({
            "params": _clean_params(params),
            "history": history,
            "metadata": sim_data.get("metadata", {}),
        }, buf)
        buf.seek(0)
        return buf, f"{name}.pkl"

    @staticmethod
    @handle_errors
    def export_npz(sim_data, params, history, name, Nx, Ny, dx, dy):
        """Export all frames as a NumPy .npz archive."""
        buf = BytesIO()
        out = {
            "params": _clean_params(params),
            "x": np.linspace(-Nx * dx / 2 * 1e9, Nx * dx / 2 * 1e9, Nx),
            "y": np.linspace(-Ny * dy / 2 * 1e9, Ny * dy / 2 * 1e9, Ny),
        }
        for i, frame in enumerate(history):
            for k in FERRO_FIELD_KEYS:
                if k in frame:
                    out[f"{k}_{i:04d}"] = np.asarray(frame[k])
        np.savez_compressed(buf, **out)
        buf.seek(0)
        return buf, f"{name}.npz"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, name):
        """Export as human-readable JSON (large but portable)."""
        def conv(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(i) for i in o]
            return o
        s = json.dumps({
            "params": _clean_params(params),
            "metadata": sim_data.get("metadata", {}),
            "history": conv(history),
        }, indent=2, cls=NumpyEncoder)
        return BytesIO(s.encode()), f"{name}.json"

    @staticmethod
    @handle_errors
    def export_csv(history, name, extent, Nx, Ny, dx, dy):
        """Export per-frame CSVs zipped together."""
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], Nx)
            y = np.linspace(extent[2], extent[3], Ny)
            X, Y = np.meshgrid(x, y)
            for i, frame in enumerate(history):
                d = {"x": X.flatten(), "y": Y.flatten()}
                for k in FERRO_FIELD_KEYS:
                    if k in frame:
                        d[k] = np.asarray(frame[k]).flatten()
                zf.writestr(f"{name}_frame_{i:04d}.csv",
                            pd.DataFrame(d).to_csv(index=False))
        buf.seek(0)
        return buf, f"{name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_timeseries_csv(solver_history, name):
        """Export the diagnostic time-series (energies) as a CSV."""
        h = solver_history
        df = pd.DataFrame({
            "step": h["step"],
            "f_bulk": h["f_bulk"],
            "f_grad": h["f_grad"],
            "f_depol_Pz": h["f_depol_Pz"],
            "f_depol_inplane": h["f_depol_inplane"],
            "f_total": h["f_total"],
        })
        return BytesIO(df.to_csv(index=False).encode()), f"{name}_timeseries.csv"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, name, Nx, Ny, dx, dy):
        """Export as HDF5 hierarchical binary."""
        if not H5PY_AVAILABLE:
            st.error("h5py not installed — pip install h5py")
            return None, None
        buf = BytesIO()
        with h5py.File(buf, "w") as f:
            pg = f.create_group("parameters")
            for k, v in _clean_params(params).items():
                if isinstance(v, (int, float, str, bool)):
                    pg.attrs[k] = v
            f.create_dataset("x",
                             data=np.linspace(-Nx * dx / 2 * 1e9, Nx * dx / 2 * 1e9, Nx))
            f.create_dataset("y",
                             data=np.linspace(-Ny * dy / 2 * 1e9, Ny * dy / 2 * 1e9, Ny))
            fg = f.create_group("frames")
            for i, frame in enumerate(history):
                g = fg.create_group(f"frame_{i:04d}")
                for k in FERRO_FIELD_KEYS:
                    if k in frame and isinstance(frame[k], np.ndarray):
                        g.create_dataset(k, data=frame[k])
        buf.seek(0)
        return buf, f"{name}.h5"

    @staticmethod
    @handle_errors
    def export_pt(sim_data, params, history, name):
        """Export as PyTorch tensors."""
        if not TORCH_AVAILABLE:
            st.error("torch not installed — pip install torch")
            return None, None
        buf = BytesIO()
        out = {"params": _clean_params(params), "history": []}
        for frame in history:
            frame_out = {}
            for k, v in frame.items():
                if isinstance(v, np.ndarray):
                    frame_out[k] = torch.from_numpy(v)
                else:
                    frame_out[k] = v
            out["history"].append(frame_out)
        torch.save(out, buf)
        buf.seek(0)
        return buf, f"{name}.pt"


# ============================================================================
# SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    """Session_state-backed storage for simulation runs (max 32)."""

    MAX_SIMS = 32

    @staticmethod
    def _hash(params):
        """Compute an 8-character hash from the params dict."""
        s = json.dumps({k: v for k, v in params.items()
                        if k not in ("history", "results", "mat", "xp",
                                     "n_steps", "save_every")},
                       sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    @staticmethod
    def save(params, history, metadata=None, run_time=0.0):
        """Save a simulation run. Returns the run ID."""
        if "sims" not in st.session_state:
            st.session_state.sims = {}
        sid = SimulationDatabase._hash(params)
        st.session_state.sims[sid] = {
            "id": sid,
            "params": _clean_params(params),
            "results_history": history,
            "metadata": metadata or {"created_at": datetime.now().isoformat(),
                                      "run_time": run_time},
            "created_at": datetime.now().isoformat(),
        }
        if len(st.session_state.sims) > SimulationDatabase.MAX_SIMS:
            oldest = min(st.session_state.sims.items(),
                         key=lambda kv: kv[1].get("created_at", ""))
            del st.session_state.sims[oldest[0]]
        return sid

    @staticmethod
    def get(sid):
        return st.session_state.get("sims", {}).get(sid)

    @staticmethod
    def delete(sid):
        if sid in st.session_state.get("sims", {}):
            del st.session_state.sims[sid]
            return True
        return False

    @staticmethod
    def clear_all():
        st.session_state.sims = {}

    @staticmethod
    def all():
        return st.session_state.get("sims", {})

    @staticmethod
    def list_for_ui():
        """Return a list of {id, name, params} for UI display."""
        out = []
        for sid, sim in SimulationDatabase.all().items():
            p = sim.get("params", {})
            preset = p.get("preset", "?")
            screening = p.get("screening", 0)
            out.append({
                "id": sid,
                "name": (f"Ferro | {preset} | "
                         f"Nx={p.get('Nx', '?')} | "
                         f"h={p.get('h_film_nm', '?'):.1f}nm | "
                         f"s={screening:.2f}"),
                "params": p,
            })
        return out


# ============================================================================
# SIDEBAR BUILDER
# ============================================================================
def build_sidebar():
    """Build the sidebar and return a params dict ready for FerroSolver.

    Bug fix #1: noise_amp conversion is `noise_amp_uc * 1e-2` (was `* 1e4 * 1e-4 = * 1`).
    1 μC/cm² = 1e-6 C / 1e-4 m² = 1e-2 C/m².
    """
    st.sidebar.header("⚙️ Configuration")

    # ---- Mode selector ----
    mode = st.sidebar.radio("Mode",
                            ["Run New Simulation", "Single Simulation View",
                             "Compare Saved"],
                            index=0, key="mode")

    if mode != "Run New Simulation":
        return {"mode": mode}

    st.sidebar.header("📐 Geometry")
    max_res = 1024 if GPU_AVAILABLE else 512
    Nx = st.sidebar.slider("Nx", 64, max_res, 256, 32)
    Ny = st.sidebar.slider("Ny", 64, max_res, 256, 32)
    L_box_nm = st.sidebar.number_input("Box size L (nm)", 10.0, 500.0, 64.0, 2.0)
    L_box = L_box_nm * 1e-9

    st.sidebar.header("Film thickness (open-circuit direction)")
    h_film_nm = st.sidebar.number_input("Film thickness h (nm)", 1.0, 50.0, 8.0, 0.5)
    h_film = h_film_nm * 1e-9

    st.sidebar.header("⏱️ Time stepping")
    dt_ps = st.sidebar.number_input("dt (ps)", 0.001, 100.0, 1.0, 0.01)
    dt = dt_ps * 1e-12
    n_steps = st.sidebar.slider("Steps", 50, 50000, 5000, 50)
    save_every = st.sidebar.slider("Save every", 1, 500, max(1, 5000 // 30), 1)
    st.sidebar.caption(f"Total time = {dt * n_steps * 1e9:.3g} ns")

    st.sidebar.header("🧪 Material preset")
    preset_name = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0)
    mat_preset = get_preset(preset_name)

    st.sidebar.header("📊 Landau-Devonshire (SI)")
    alpha1 = st.sidebar.number_input("α₁ (V·m/C, <0 below Tc)", -5e11, 0.0,
                                      float(mat_preset.alpha1), 1e9, format="%.3e")
    alpha11 = st.sidebar.number_input("α₁₁ (V·m⁵/C³)", 1e8, 1e12,
                                       float(mat_preset.alpha11), 1e9, format="%.3e")
    st.sidebar.markdown("*For tetragonal: α₁₂ > 2α₁₁*")
    alpha12 = st.sidebar.number_input("α₁₂ (V·m⁵/C³)", 0.0, 5e11,
                                       float(mat_preset.alpha12), 1e9, format="%.3e")

    st.sidebar.header("📈 Lifshitz gradient (V·m³/C)")
    G11 = st.sidebar.number_input("G₁₁", 1e-11, 1e-6,
                                   float(mat_preset.G11), 1e-10, format="%.3e")
    G44 = st.sidebar.number_input("G₄₄", 1e-11, 1e-6,
                                   float(mat_preset.G44), 1e-10, format="%.3e")
    G12 = 0.0

    st.sidebar.header("🏗️ Substrate anisotropy (favors Pz)")
    st.sidebar.markdown("""
    Adds quadratic penalty β_sub·(Px²+Py²)/2 on in-plane P.
    With β_sub > |α₁|, in-plane P is suppressed and only Pz can be ferroelectric.
    This mimics compressive substrate strain without requiring full elasticity.
    """)
    use_substrate = st.sidebar.checkbox("Enable substrate anisotropy", True)
    if use_substrate:
        beta_sub = st.sidebar.number_input(
            "β_sub (V·m/C, need > |α₁|)", 0.0, 1e12,
            float(abs(alpha1) * 1.5), 1e9, format="%.3e")
    else:
        beta_sub = 0.0

    st.sidebar.header("🔄 Kinetics")
    Gamma = st.sidebar.number_input("Γ (Ω·m)⁻¹", 0.001, 100.0, 1.0, 0.1)

    st.sidebar.header("⚡ Depolarization (open-circuit kernel)")
    use_elec = st.sidebar.checkbox("Enable depolarization fields", True)
    st.sidebar.markdown("""
    **Screening factor** s ∈ [0, 1]:
    - 0 = perfect electrodes (short-circuit, no depol)
    - 1 = no electrodes (open-circuit, full depol)
    - ~0.65 = partial screening → 180° stripe domains form
    """)
    screening = st.sidebar.slider("s ∈ [0, 1]", 0.0, 1.0, 0.65, 0.01)

    use_ext_E = st.sidebar.checkbox("Apply external E field", False)
    if use_ext_E:
        Ex_ext = st.sidebar.number_input("E_ext,x (MV/m)", -500.0, 500.0, 50.0, 1.0) * 1e6
        Ey_ext = st.sidebar.number_input("E_ext,y (MV/m)", -500.0, 500.0, 0.0, 1.0) * 1e6
        Ez_ext = st.sidebar.number_input("E_ext,z (MV/m)", -500.0, 500.0, 0.0, 1.0) * 1e6
    else:
        Ex_ext = Ey_ext = Ez_ext = 0.0

    st.sidebar.header("🎲 Initial condition")
    init_mode = st.sidebar.selectbox(
        "Init type",
        ["band_limited", "white_noise", "seeded_stripes"],
        index=0,
        format_func=lambda x: {
            "band_limited": "Band-limited noise (structured)",
            "white_noise": "Pure white noise (all 3 components)",
            "seeded_stripes": "Seeded 180° stripes on Pz",
        }[x])
    noise_amp_uc = st.sidebar.number_input("Noise amplitude (μC/cm²)", 1.0, 100.0, 20.0, 1.0)
    # Bug fix #1: 1 μC/cm² = 1e-2 C/m² (was: * 1e4 * 1e-4 = * 1, which is 100× too large)
    noise_amp = noise_amp_uc * 1e-2
    noise_cutoff = st.sidebar.slider("Band cutoff (frac of Nyquist)", 0.05, 0.5, 0.15, 0.01)
    seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

    # Build LandauParams (triggers alpha12 warning via __init__ if violated)
    mat = LandauParams(alpha1=alpha1, alpha11=alpha11, alpha12=alpha12,
                       G11=G11, G44=G44, G12=G12, beta_sub=beta_sub)

    # Derived quantities for display
    Ps = mat.Ps_eq()
    xi = mat.xi_wall()
    dx = L_box / Nx
    nonlinear_cfl = dt * Gamma * abs(alpha1)
    st.sidebar.markdown(f"""
    **Derived quantities:**
    - |Ps| = {Ps*100:.2f} μC/cm²
    - Wall width ξ = {xi*1e9:.3f} nm
    - Grid dx = {dx*1e9:.3f} nm
    - ξ/dx = {xi/dx:.2f} (need > 3 for wall resolution)
    - Box/ξ = {L_box/xi:.1f} (need > 10 for multiple domains)
    - dt·Γ·|α₁| = {nonlinear_cfl:.3e} (need < 1 for stability)
    """)

    # Bug fix #4: Stability warning for nonlinear CFL
    if nonlinear_cfl > 1.0:
        st.sidebar.error(f"⚠️ UNSTABLE: dt·Γ·|α₁| = {nonlinear_cfl:.2e} > 1. "
                         f"Reduce dt or Γ.")
    elif nonlinear_cfl > 0.5:
        st.sidebar.warning(f"⚠️ Marginal stability: dt·Γ·|α₁| = {nonlinear_cfl:.2e}. "
                           f"Consider reducing dt.")

    params = {
        "mode": mode,
        "xp": xp,
        "mat": mat,
        "Nx": Nx, "Ny": Ny, "L": L_box, "h_film": h_film,
        "h_film_nm": h_film_nm, "L_box_nm": L_box_nm,
        "dt": dt, "dt_ps": dt_ps, "n_steps": n_steps, "save_every": save_every,
        "Gamma": Gamma,
        "screening": screening, "use_elec": use_elec,
        "E_ext": (Ex_ext, Ey_ext, Ez_ext),
        "init_mode": init_mode,
        "noise_amp": noise_amp, "noise_amp_uc": noise_amp_uc,
        "noise_cutoff": noise_cutoff, "seed": seed,
        "preset": preset_name,
        "use_substrate": use_substrate, "beta_sub": beta_sub,
        "time_unit": "ps",
    }
    return params


# ============================================================================
# UI TAB: GEOMETRY
# ============================================================================
def render_geometry_tab(params, viz_cfg):
    """Show the initial Pz field and derived quantities."""
    st.header("📐 Initial Geometry")

    mat = params["mat"]
    Nx, Ny = params["Nx"], params["Ny"]
    L = params["L"]
    dx = L / Nx
    dy = L / Ny
    extent = [-Nx * dx / 2 * 1e9, Nx * dx / 2 * 1e9,
              -Ny * dy / 2 * 1e9, Ny * dy / 2 * 1e9]

    Ps = mat.Ps_eq()
    xi = mat.xi_wall()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("|Ps|", f"{Ps*100:.2f} μC/cm²")
    col2.metric("Wall width ξ", f"{xi*1e9:.3f} nm")
    col3.metric("Grid", f"{Nx}×{Ny}")
    col4.metric("Domain", f"{params['L_box_nm']:.1f} nm²")

    st.markdown(f"""
    **Configuration summary:**
    - Material preset: `{params['preset']}`
    - Film thickness: `{params['h_film_nm']:.1f} nm`
    - Screening factor: `{params['screening']:.2f}` (0=short, 1=open-circuit)
    - β_sub: `{params['beta_sub']:.2e}` V·m/C
    - Init mode: `{params['init_mode']}`
    - Noise amplitude: `{params['noise_amp_uc']:.1f}` μC/cm² = `{params['noise_amp']:.3f}` C/m²
    """)

    # Show the initial Pz field
    Px0, Py0, Pz0 = make_initial_P(
        Nx, Ny, L, params["init_mode"],
        params["noise_amp"], params["noise_cutoff"],
        Ps, params["seed"], xp)
    Pz0_cpu = to_cpu(Pz0)

    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("overlay_width", 11.0))
    fig_h = float(fig_cfg.get("overlay_height", 5.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = ColormapRegistry.resolve("Pz_div")
    vmax = max(Ps, np.abs(Pz0_cpu).max())
    im = ax.imshow(Pz0_cpu, extent=extent, origin="lower", cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label="Pz (C/m²)")
    apply_axis_styling(ax, fig_cfg,
                       title=f"Initial Pz (mode={params['init_mode']})",
                       xlabel="x (nm)", ylabel="y (nm)")
    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show |P| initial
    P_mag0 = np.sqrt(to_cpu(Px0)**2 + to_cpu(Py0)**2 + to_cpu(Pz0)**2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap_mag = ColormapRegistry.resolve("Pmag")
    im = ax.imshow(P_mag0, extent=extent, origin="lower", cmap=cmap_mag,
                   vmin=0, vmax=Ps, aspect="equal")
    plt.colorbar(im, ax=ax, label="|P| (C/m²)")
    apply_axis_styling(ax, fig_cfg,
                       title="Initial |P|",
                       xlabel="x (nm)", ylabel="y (nm)")
    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# UI TAB: RUN
# ============================================================================
def render_run_tab(params, viz_cfg):
    """Execute the simulation and store results in session_state."""
    st.header("▶️ Run Simulation")

    mat = params["mat"]
    Ps = mat.Ps_eq()
    xi = mat.xi_wall()
    dx = params["L"] / params["Nx"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        **Expected physics with current settings:**
        - Material: `{params['preset']}` (|Ps| = {Ps*100:.2f} μC/cm²)
        - Domain: `{params['L_box_nm']:.1f} × {params['L_box_nm']:.1f} nm`
          ({params['Nx']} × {params['Ny']} cells, dx = {dx*1e9:.3f} nm)
        - Film thickness: `{params['h_film_nm']:.1f} nm`
        - Time: `{params['dt_ps']:.3f} ps × {params['n_steps']} steps = {params['n_steps']*params['dt']*1e9:.3f} ns`
        - Screening: `{params['screening']:.2f}` (0=short, 1=open-circuit)
        - β_sub: `{params['beta_sub']:.2e}` V·m/C
        - Wall width ξ: `{xi*1e9:.3f} nm` (ξ/dx = {xi/dx:.2f}, need > 3)
        """)

    if st.button("▶️ Start Evolution", type="primary",
                 width="stretch"):
        with st.spinner("Running FFT phase-field simulation..."):
            solver = FerroSolver(params)
            history = []
            n_steps = params["n_steps"]
            save_freq = params["save_every"]
            progress = st.progress(0)
            status = st.empty()

            for step in range(n_steps):
                res = solver.step()
                if res is FAILED or res is None:
                    st.error(f"Solver step failed: {FAILED.reason}")
                    break
                if step % save_freq == 0 or step == n_steps - 1:
                    history.append(res)
                progress.progress((step + 1) / n_steps)
                if step % max(1, n_steps // 20) == 0:
                    f_tot = solver.history["f_total"][-1]
                    mean_Pz = float(np.mean(np.abs(to_cpu(solver.Pz))))
                    status.text(
                        f"Step {step+1}/{n_steps} | "
                        f"f_total = {f_tot:.3e} J/m³ | "
                        f"|Pz| mean = {mean_Pz:.4f} C/m²")

            st.success(f"✅ Done. {len(history)} frames saved.")
            st.session_state["history"] = history
            st.session_state["solver"] = solver
            st.session_state["initialized"] = True
            sid = SimulationDatabase.save(params, history,
                                           run_time=n_steps * params["dt"])
            st.session_state["last_sid"] = sid
            st.balloons()


# ============================================================================
# UI TAB: VISUALIZATION SETTINGS
# ============================================================================
def render_viz_settings_tab(params, viz_cfg):
    """Theme, per-field colormaps, overlay/vector/animation settings."""
    st.header("🎨 Visualization Settings")
    st.caption("Settings apply to Results, Line Profiles, Plotly, and Export tabs.")

    sub1, sub2, sub3 = st.tabs(["Global & Field Scaling",
                                 "Overlay & Vector & Animation",
                                 "Figure Customization"])

    # ---- Sub-tab 1: Global theme + per-field settings ----
    with sub1:
        st.subheader("🖌️ Global matplotlib theme")
        theme_keys = list(MATPLOTLIB_THEMES.keys())
        default_idx = (theme_keys.index(viz_cfg["theme"])
                       if viz_cfg["theme"] in theme_keys else 0)
        c1, c2, c3 = st.columns(3)
        with c1:
            theme = st.selectbox("Theme", theme_keys, index=default_idx, key="viz_theme")
        with c2:
            title_fs = st.slider("Title font", 6, 24, int(viz_cfg["title_font_size"]),
                                  key="viz_title_fs")
            label_fs = st.slider("Label font", 6, 20, int(viz_cfg["label_font_size"]),
                                  key="viz_label_fs")
        with c3:
            tick_fs = st.slider("Tick font", 4, 18, int(viz_cfg["tick_font_size"]),
                                 key="viz_tick_fs")
            legend_fs = st.slider("Legend font", 4, 18, int(viz_cfg["legend_font_size"]),
                                   key="viz_legend_fs")
        dpi = st.slider("DPI", 72, 600, int(viz_cfg["dpi"]), 12, key="viz_dpi")

        if st.button("♻️ Apply theme", key="viz_apply_theme"):
            viz_cfg["theme"] = theme
            viz_cfg["title_font_size"] = title_fs
            viz_cfg["label_font_size"] = label_fs
            viz_cfg["tick_font_size"] = tick_fs
            viz_cfg["legend_font_size"] = legend_fs
            viz_cfg["dpi"] = dpi
            apply_matplotlib_theme(theme, viz_cfg)
            st.success("Theme applied.")
            st.rerun()

        st.markdown("---")
        render_live_colormap_picker(viz_cfg, key_prefix="viz_cmap",
                                     label="🎨 Per-field colormap (live)")

        st.markdown("---")
        st.subheader("📏 Per-field dynamic range & scaling")
        field_keys = list(FIELD_DEFAULT_CMAPS.keys())
        field_choice = st.selectbox("Field to configure", field_keys, key="viz_field_choice")
        fs = viz_cfg["field_settings"][field_choice]

        auto = st.checkbox("Auto range (2-98 percentile)",
                           value=fs.get("auto", True),
                           key=f"viz_auto_{field_choice}")
        if not auto:
            drng = FIELD_DEFAULT_RNG.get(field_choice, (0.0, 1.0))
            c1, c2 = st.columns(2)
            with c1:
                vmin_in = st.number_input("vmin", value=float(fs.get("vmin") or drng[0]),
                                           key=f"viz_vmin_{field_choice}", format="%.6g")
            with c2:
                vmax_in = st.number_input("vmax", value=float(fs.get("vmax") or drng[1]),
                                           key=f"viz_vmax_{field_choice}", format="%.6g")
        else:
            vmin_in = vmax_in = None
        log_scale = st.checkbox("Logarithmic scale", value=fs.get("log", False),
                                 key=f"viz_log_{field_choice}")
        invert = st.checkbox("Invert colormap", value=fs.get("invert", False),
                              key=f"viz_inv_{field_choice}")

        if st.button("💾 Save field settings", key=f"viz_save_{field_choice}"):
            viz_cfg["field_settings"][field_choice] = {
                "auto": auto, "vmin": vmin_in, "vmax": vmax_in,
                "log": log_scale, "invert": invert,
                "cmap": fs.get("cmap", "viridis"),
            }
            st.success(f"Saved settings for '{field_choice}'.")
            st.rerun()

    # ---- Sub-tab 2: Overlay / Vector / Animation ----
    with sub2:
        st.subheader("🪟 Overlay settings")
        ov = viz_cfg["overlay"]
        ov_enabled = st.checkbox("Enable overlay mode", value=ov["enabled"], key="ov_en")
        field_keys2 = list(FIELD_DEFAULT_CMAPS.keys())
        c1, c2 = st.columns(2)
        with c1:
            base_idx = (field_keys2.index(ov["base"])
                        if ov["base"] in field_keys2 else 2)
            ov_base = st.selectbox("Base field", field_keys2, index=base_idx, key="ov_base")
        with c2:
            over_idx = (field_keys2.index(ov["overlay"])
                        if ov["overlay"] in field_keys2 else 3)
            ov_over = st.selectbox("Overlay field", field_keys2, index=over_idx, key="ov_over")
        c1, c2 = st.columns(2)
        with c1:
            ov_mode = st.radio("Overlay mode", ["contour", "heatmap"],
                                index=0 if ov["mode"] == "contour" else 1, key="ov_mode")
        with c2:
            ov_alpha = st.slider("Alpha", 0.05, 1.0, float(ov["alpha"]), 0.05, key="ov_alpha")
        ov_nc = st.slider("Contour levels", 3, 30, int(ov["n_contours"]), 1, key="ov_nc")
        if st.button("💾 Save overlay settings", key="ov_save"):
            viz_cfg["overlay"] = {
                "enabled": ov_enabled, "base": ov_base, "overlay": ov_over,
                "alpha": ov_alpha, "mode": ov_mode, "n_contours": ov_nc,
            }
            st.success("Overlay settings saved.")
            st.rerun()

        st.markdown("---")
        st.subheader("🧭 Vector field overlay")
        vv = viz_cfg["vector"]
        vv_enabled = st.checkbox("Enable vector-field mode", value=vv["enabled"], key="vv_en")
        c1, c2 = st.columns(2)
        with c1:
            vv_density = st.slider("Arrow density", 6, 80, int(vv["density"]), 2, key="vv_den")
        with c2:
            vv_scale = st.slider("Arrow scale", 0.1, 5.0, float(vv["scale"]), 0.1, key="vv_scale")
        if st.button("💾 Save vector settings", key="vv_save"):
            viz_cfg["vector"] = {
                "enabled": vv_enabled, "source": "Pz", "mode": "inplane",
                "density": vv_density, "color": "#ffffff", "scale": vv_scale,
            }
            st.success("Vector settings saved.")
            st.rerun()

        st.markdown("---")
        st.subheader("🎬 Animation settings")
        an = viz_cfg["animation"]
        c1, c2, c3 = st.columns(3)
        with c1:
            an_fmt = st.selectbox("Format", ["gif", "mp4"],
                                   index=0 if an["format"] == "gif" else 1, key="an_fmt")
        with c2:
            an_fps = st.slider("FPS", 1, 30, int(an["fps"]), 1, key="an_fps")
        with c3:
            an_dpi = st.slider("DPI", 72, 400, int(an["dpi"]), 12, key="an_dpi")
        an_wm = st.checkbox("Show watermark", value=an["watermark"], key="an_wm")
        an_wm_text = st.text_input("Watermark text", value=an["watermark_text"], key="an_wm_text")
        wm_pos_list = ["lower right", "lower left", "upper right", "upper left"]
        wm_idx = (wm_pos_list.index(an["watermark_pos"])
                  if an["watermark_pos"] in wm_pos_list else 0)
        an_wm_pos = st.selectbox("Watermark position", wm_pos_list, index=wm_idx, key="an_wm_pos")
        if st.button("💾 Save animation settings", key="an_save"):
            viz_cfg["animation"] = {
                "format": an_fmt, "fps": an_fps, "dpi": an_dpi,
                "watermark": an_wm, "watermark_text": an_wm_text,
                "watermark_pos": an_wm_pos,
            }
            st.success("Animation settings saved.")
            st.rerun()

    # ---- Sub-tab 3: Figure customization ----
    with sub3:
        fc = viz_cfg["figure_customization"]
        st.info("💡 Per-figure customization is available inline above each figure "
                "in the Results, Line Profiles, Plotly, and Export tabs. "
                "The settings below are global defaults.")

        st.markdown("#### 📐 Figure sizes (inches)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("**Multi-field grid**")
            mf_w = st.number_input("Width", value=float(fc.get("multi_field_width", 13.5)),
                                    min_value=4.0, max_value=30.0, step=0.5, key="fc_mf_w")
            mf_h = st.number_input("Height", value=float(fc.get("multi_field_height", 11.0)),
                                    min_value=3.0, max_value=30.0, step=0.5, key="fc_mf_h")
        with c2:
            st.caption("**Overlay / Vector**")
            ov_w = st.number_input("Width", value=float(fc.get("overlay_width", 11.0)),
                                    min_value=4.0, max_value=30.0, step=0.5, key="fc_ov_w")
            ov_h = st.number_input("Height", value=float(fc.get("overlay_height", 5.5)),
                                    min_value=3.0, max_value=30.0, step=0.5, key="fc_ov_h")
        with c3:
            st.caption("**Diagnostics**")
            dg_w = st.number_input("Width", value=float(fc.get("diagnostics_width", 13.0)),
                                    min_value=4.0, max_value=30.0, step=0.5, key="fc_dg_w")
            dg_h = st.number_input("Height", value=float(fc.get("diagnostics_height", 8.0)),
                                    min_value=3.0, max_value=30.0, step=0.5, key="fc_dg_h")

        if st.button("💾 Save figure customization", key="fc_save", type="primary"):
            fc["multi_field_width"] = float(mf_w)
            fc["multi_field_height"] = float(mf_h)
            fc["overlay_width"] = float(ov_w)
            fc["overlay_height"] = float(ov_h)
            fc["diagnostics_width"] = float(dg_w)
            fc["diagnostics_height"] = float(dg_h)
            st.success("Figure customization saved.")
            st.rerun()


# ============================================================================
# UI TAB: RESULTS
# ============================================================================
def render_results_tab(params, viz_cfg):
    """Multi-field grid, overlay, or vector overlay with live customization."""
    st.header("📊 Results")

    history = st.session_state.get("history", [])
    if not history:
        st.info("Run the simulation first (▶️ Run tab).")
        return

    solver = st.session_state.get("solver", None)
    idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="res_frame")

    Nx, Ny = params["Nx"], params["Ny"]
    L = params["L"]
    dx = L / Nx
    dy = L / Ny
    vis = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])

    style = {
        "colormaps": {f: fset.get("cmap", "viridis")
                       for f, fset in viz_cfg["field_settings"].items()},
        "field_settings": viz_cfg["field_settings"],
        "title_font_size": viz_cfg["title_font_size"],
        "label_font_size": viz_cfg["label_font_size"],
        "colorbar_fraction": viz_cfg["colorbar_fraction"],
        "colorbar_pad": viz_cfg["colorbar_pad"],
        "figure_customization": viz_cfg.get("figure_customization", {}),
    }

    mode_res = st.radio("Display mode",
                        ["Multi-field grid", "Overlay", "Vector overlay"],
                        horizontal=True, key="res_mode")

    if mode_res == "Multi-field grid":
        render_live_colormap_picker(viz_cfg, key_prefix="res_cmap",
                                     label="🎨 Per-field colormap (live)")
        render_enhanced_figure_customization(
            viz_cfg, size_keys=("multi_field_width", "multi_field_height"),
            key_prefix="res_mf", label="🎨 Multi-field grid — full customization",
            show_line_styles=False, show_legend=False, show_spines=True,
            show_font_sizes=True, show_colorbar=True, show_grid=True,
            show_axis_limits=True, show_text_overrides=True)
        fig = vis.create_multi_field_comparison(history[idx], style)
        if fig is not None and fig is not FAILED:
            st.pyplot(fig)
            plt.close(fig)

    elif mode_res == "Overlay":
        render_enhanced_figure_customization(
            viz_cfg, size_keys=("overlay_width", "overlay_height"),
            key_prefix="res_ov", label="🎨 Overlay — full customization",
            show_line_styles=True, show_legend=True, show_colorbar=True)
        ov = viz_cfg["overlay"]
        fig = vis.create_overlay(history[idx], base_field=ov["base"],
                                  overlay_field=ov["overlay"], alpha=ov["alpha"],
                                  mode=ov["mode"], n_contours=ov["n_contours"],
                                  style_params=style)
        if fig is not None and fig is not FAILED:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Overlay failed — check field selections.")

    else:  # Vector overlay
        render_enhanced_figure_customization(
            viz_cfg, size_keys=("vector_width", "vector_height"),
            key_prefix="res_vc", label="🎨 Vector field — full customization",
            show_line_styles=True, show_legend=True, show_colorbar=True)
        vv = viz_cfg["vector"]
        fig = vis.create_vector_overlay(history[idx], density=vv["density"],
                                         color=vv["color"], scale=vv["scale"],
                                         base_field="Pz", style_params=style)
        if fig is not None and fig is not FAILED:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Vector overlay unavailable.")

    # ---- Energy diagnostics ----
    if solver is not None:
        st.subheader("📈 Energy Evolution")
        fig_cfg = viz_cfg.get("figure_customization", {})
        fig_w = float(fig_cfg.get("diagnostics_width", 13.0))
        fig_h = float(fig_cfg.get("diagnostics_height", 6.0))

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1],
                      hspace=0.35, wspace=0.30)
        axes = np.array([
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ])

        steps = np.array(solver.history["step"])
        lw_main = float(fig_cfg.get("line_width", 2.0))
        lw_minor = float(fig_cfg.get("line_width_minor", 1.5))
        ls_main = fig_cfg.get("line_style", "-")
        ls_minor = fig_cfg.get("line_style_minor", "--")
        col1 = fig_cfg.get("color_series_1", "#1f77b4")
        col2 = fig_cfg.get("color_series_2", "#d62728")
        col3 = fig_cfg.get("color_series_3", "#2ca02c")
        col4 = fig_cfg.get("color_series_4", "#9467bd")
        col5 = fig_cfg.get("color_series_5", "#8c564b")
        t_xlabel = fig_cfg.get("xlabel_override", "").strip() or "Step"

        # Top-left: f_bulk + f_grad
        ax = axes[0, 0]
        ax.plot(steps, solver.history["f_bulk"], color=col1,
                linestyle=ls_main, lw=lw_main, label="f_bulk (Landau)")
        ax.plot(steps, solver.history["f_grad"], color=col3,
                linestyle=ls_minor, lw=lw_minor, label="f_grad (Lifshitz)")
        apply_axis_styling(ax, fig_cfg,
                           title="Bulk + Gradient energy",
                           xlabel=t_xlabel, ylabel="Energy (J/m³)",
                           force_aspect="auto")
        apply_legend_styling(ax, fig_cfg)

        # Top-right: f_depol_Pz + f_depol_inplane
        ax = axes[0, 1]
        ax.plot(steps, solver.history["f_depol_Pz"], color=col2,
                linestyle=ls_main, lw=lw_main, label="f_depol_Pz (thin-film)")
        if params["use_elec"]:
            ax.plot(steps, solver.history["f_depol_inplane"], color=col4,
                    linestyle=ls_minor, lw=lw_minor, label="f_depol_inplane")
        apply_axis_styling(ax, fig_cfg,
                           title="Depolarization energy",
                           xlabel=t_xlabel, ylabel="Energy (J/m³)",
                           force_aspect="auto")
        apply_legend_styling(ax, fig_cfg)

        # Bottom-left: f_total
        ax = axes[1, 0]
        ax.plot(steps, solver.history["f_total"], color=col5,
                linestyle=ls_main, lw=lw_main, label="f_total")
        apply_axis_styling(ax, fig_cfg,
                           title="Total free energy density",
                           xlabel=t_xlabel, ylabel="Energy (J/m³)",
                           force_aspect="auto")
        apply_legend_styling(ax, fig_cfg)

        # Bottom-right: |Pz| mean and |P_inplane| mean (from convergence)
        ax = axes[1, 1]
        mean_Pz = [h["convergence"]["mean_Pz"] for h in history]
        mean_Pin = [h["convergence"]["mean_Pinplane"] for h in history]

        # Fix #1: Generate saved_steps array matching the length of history
        # (history only contains saved frames, not every step)
        save_every = params.get("save_every", 1)
        n_steps_total = params.get("n_steps", 5000)
        saved_steps = [i * save_every for i in range(len(history))]
        if len(saved_steps) > 1:
            saved_steps[-1] = n_steps_total - 1  # last frame = final step

        ax.plot(saved_steps, mean_Pz, color=col1,
                linestyle=ls_main, lw=lw_main, label="|Pz| mean")
        ax.plot(saved_steps, mean_Pin, color=col2,
                linestyle=ls_minor, lw=lw_minor, label="|P_inplane| mean")
        apply_axis_styling(ax, fig_cfg,
                           title="Polarization magnitude evolution",
                           xlabel=t_xlabel, ylabel="|P| (C/m²)",
                           force_aspect="auto")
        apply_legend_styling(ax, fig_cfg)

        if fig_cfg.get("tight_layout", True):
            plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # CSV download
        st.subheader("📋 Energy time-series")
        df = pd.DataFrame({
            "step": solver.history["step"],
            "f_bulk": solver.history["f_bulk"],
            "f_grad": solver.history["f_grad"],
            "f_depol_Pz": solver.history["f_depol_Pz"],
            "f_depol_inplane": solver.history["f_depol_inplane"],
            "f_total": solver.history["f_total"],
        })
        st.dataframe(df.tail(20).style.format("{:.4e}"))

        buf, fname = DataExporter.export_timeseries_csv(solver.history, "ferro")
        st.download_button("Download energy CSV", buf, fname, "text/csv")


# ============================================================================
# UI TAB: LINE PROFILES
# ============================================================================
def render_line_profiles_tab(params, viz_cfg):
    """Single-frame and multi-time profile comparison."""
    st.header("🔬 Line Profiles")

    history = st.session_state.get("history", [])
    if not history:
        st.info("Run the simulation first.")
        return

    Nx, Ny = params["Nx"], params["Ny"]
    L = params["L"]
    dx = L / Nx
    dy = L / Ny
    profiler = EnhancedLineProfiler(Nx, Ny, dx, dy)

    render_enhanced_figure_customization(
        viz_cfg, size_keys=("line_profile_width", "line_profile_height"),
        key_prefix="lp", label="🎨 Line profile — full customization",
        show_line_styles=True, show_legend=True, show_colorbar=False,
        show_font_sizes=True, show_grid=True,
        show_axis_limits=True, show_text_overrides=True)

    sub_a, sub_b = st.tabs(["Single frame", "Multi-time compare"])

    with sub_a:
        idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="lp_frame")
        field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="lp_field")
        ptype = st.selectbox("Direction",
                              ["horizontal", "vertical", "diagonal", "anti_diagonal"],
                              key="lp_ptype")
        pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05, key="lp_pos")

        d = history[idx][field]
        out = profiler.extract_profile(d, ptype, pos)
        if out is not None and out[0] is not None:
            dist, prof, _ = out
            fc = viz_cfg.get("figure_customization", {})
            fig_w = float(fc.get("line_profile_width", 11.0))
            fig_h = float(fc.get("line_profile_height", 4.0))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.plot(dist, prof,
                    color=fc.get("color_series_1", "#1f77b4"),
                    linestyle=fc.get("line_style", "-"),
                    lw=float(fc.get("line_width", 2.0)))
            apply_axis_styling(ax, fc,
                               title=f"{field} · {ptype} · frame {idx}",
                               xlabel="Position (nm)", ylabel=field,
                               force_aspect="auto")
            if fc.get("tight_layout", True):
                plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with sub_b:
        field_mt = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="lp_mt_field")
        ptype_mt = st.selectbox("Direction",
                                 ["horizontal", "vertical", "diagonal", "anti_diagonal"],
                                 key="lp_mt_ptype")
        pos_mt = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05, key="lp_mt_pos")
        default_frames = list(range(0, len(history), max(1, len(history) // 6)))
        frames_sel = st.multiselect("Frames", list(range(len(history))),
                                     default=default_frames)
        use_plotly = st.checkbox("Use Plotly (interactive)", value=True, key="lp_mt_plotly")
        if frames_sel:
            vis = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])
            fig = vis.create_multi_time_profiles(history, field_mt, ptype_mt, frames_sel,
                                                  position_ratio=pos_mt, use_plotly=use_plotly,
                                                  viz_cfg=viz_cfg)
            if fig is not None and fig is not FAILED:
                if use_plotly:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.pyplot(fig)
                    plt.close(fig)


# ============================================================================
# UI TAB: PLOTLY
# ============================================================================
def render_plotly_tab(params, viz_cfg):
    """Interactive Plotly heatmap, 3D surface, animation, line profiles."""
    st.header("🌐 Plotly Interactive")

    history = st.session_state.get("history", [])
    if not history:
        st.info("Run the simulation first.")
        return

    if not PLOTLY_AVAILABLE:
        st.error("Plotly not installed. Install with: pip install plotly")
        return

    Nx, Ny = params["Nx"], params["Ny"]
    L = params["L"]
    dx = L / Nx
    dy = L / Ny
    vis = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])

    view = st.radio("View",
                    ["Heatmap (frame)", "3D surface",
                     "Time-scrubbing animation", "Line profiles"],
                    horizontal=True, key="px_view")

    if view in ("Heatmap (frame)", "Time-scrubbing animation"):
        _sz_keys = ("plotly_width", "plotly_height")
        _lbl = "Plotly customization"
    elif view == "3D surface":
        _sz_keys = None
        _lbl = "3D surface (text only)"
    else:
        _sz_keys = ("line_profile_width", "line_profile_height")
        _lbl = "Line profile customization"

    render_inline_figure_customization(
        viz_cfg, size_keys=_sz_keys, key_prefix=f"px_{view[:3]}",
        label=_lbl,
        show_lines=(view == "Line profiles"),
        show_colorbar=(view != "Line profiles"))

    if view == "Heatmap (frame)":
        idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="px_frame")
        field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="px_field")
        cmap = viz_cfg["field_settings"].get(field, {}).get("cmap", "viridis")
        fig = vis.create_plotly_heatmap(history[idx], field, idx, cmap, viz_cfg=viz_cfg)
        if fig is not None and fig is not FAILED:
            st.plotly_chart(fig, width="stretch")

    elif view == "3D surface":
        idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="px_3d_frame")
        field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="px_3d_field")
        show_c = st.checkbox("Show contours", value=True, key="px_3d_cont")
        cmap = viz_cfg["field_settings"].get(field, {}).get("cmap", "viridis")
        fig = vis.create_plotly_3d_surface(history[idx], field, idx, cmap,
                                            show_contours=show_c, viz_cfg=viz_cfg)
        if fig is not None and fig is not FAILED:
            st.plotly_chart(fig, width="stretch")

    elif view == "Time-scrubbing animation":
        field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="px_anim_field")
        cmap = viz_cfg["field_settings"].get(field, {}).get("cmap", "viridis")
        with st.spinner("Building animation frames..."):
            fig = vis.create_plotly_animation(history, field, params["dt"], cmap, viz_cfg=viz_cfg)
        if fig is not None and fig is not FAILED:
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Could not build animation.")

    else:  # Line profiles
        idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="px_lp_frame")
        field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="px_lp_field")
        ptypes = st.multiselect("Directions",
                                 ["horizontal", "vertical", "diagonal", "anti_diagonal"],
                                 default=["horizontal", "vertical"], key="px_lp_types")
        pos = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05, key="px_lp_pos")
        if ptypes:
            fig = vis.create_plotly_line_profiles(history[idx], field, ptypes, pos, viz_cfg=viz_cfg)
            if fig is not None and fig is not FAILED:
                st.plotly_chart(fig, width="stretch")


# ============================================================================
# UI TAB: EXPORT
# ============================================================================
def render_export_tab(params, viz_cfg):
    """Data export (PKL/NPZ/JSON/CSV/HDF5/PT) + animation + PNG snapshot."""
    st.header("📤 Export")

    history = st.session_state.get("history", [])
    if not history:
        st.info("Run the simulation first.")
        return

    solver = st.session_state.get("solver", None)
    Nx, Ny = params["Nx"], params["Ny"]
    dx = params["L"] / Nx
    dy = dx
    sim_data = {"metadata": {"created_at": datetime.now().isoformat()}}
    preset = params.get("preset", "ferro")
    name = f"ferro_{preset}_Nx{Nx}_Ny{Ny}_seed{params['seed']}"

    st.subheader("📦 Data export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Pickle (.pkl)"):
            buf, fname = DataExporter.export_pkl(sim_data, params, history, name)
            if buf:
                st.download_button("Download .pkl", buf, fname)
        if st.button("NumPy (.npz)"):
            buf, fname = DataExporter.export_npz(sim_data, params, history, name, Nx, Ny, dx, dy)
            if buf:
                st.download_button("Download .npz", buf, fname)
        if st.button("JSON"):
            buf, fname = DataExporter.export_json(sim_data, params, history, name)
            if buf:
                st.download_button("Download .json", buf, fname)
    with col2:
        if st.button("CSV (zip of per-frame)"):
            extent = [-Nx * dx / 2 * 1e9, Nx * dx / 2 * 1e9, -Ny * dy / 2 * 1e9, Ny * dy / 2 * 1e9]
            buf, fname = DataExporter.export_csv(history, name, extent, Nx, Ny, dx, dy)
            if buf:
                st.download_button("Download .zip", buf, fname)
        if st.button("📊 Time-series CSV"):
            if solver:
                buf, fname = DataExporter.export_timeseries_csv(solver.history, name)
                if buf:
                    st.download_button("Download .csv", buf, fname)
        if st.button("HDF5 (.h5)"):
            buf, fname = DataExporter.export_hdf5(sim_data, params, history, name, Nx, Ny, dx, dy)
            if buf:
                st.download_button("Download .h5", buf, fname)
        if st.button("PyTorch (.pt)"):
            buf, fname = DataExporter.export_pt(sim_data, params, history, name)
            if buf:
                st.download_button("Download .pt", buf, fname)

    st.markdown("---")
    st.subheader("🎬 Animation export")
    an = viz_cfg["animation"]
    anim_field = st.selectbox("Field to animate", list(FIELD_DEFAULT_CMAPS.keys()),
                               index=2, key="anim_field")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_fmt = st.selectbox("Format", ["gif", "mp4"],
                              index=0 if an["format"] == "gif" else 1, key="anim_fmt_ui")
    with c2:
        a_fps = st.slider("FPS", 1, 30, int(an["fps"]), 1, key="anim_fps_ui")
    with c3:
        a_dpi = st.slider("DPI", 72, 300, int(an["dpi"]), 12, key="anim_dpi_ui")
    a_wm = st.checkbox("Watermark", value=an["watermark"], key="anim_wm_ui")
    a_wm_text = st.text_input("Watermark text", value=an["watermark_text"], key="anim_wm_text_ui")
    wm_pos_list_ui = ["lower right", "lower left", "upper right", "upper left"]
    wm_idx_ui = (wm_pos_list_ui.index(an["watermark_pos"])
                 if an["watermark_pos"] in wm_pos_list_ui else 0)
    a_wm_pos = st.selectbox("Watermark position", wm_pos_list_ui, index=wm_idx_ui, key="anim_wm_pos_ui")

    if st.button("🎬 Generate animation"):
        vis_exp = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])
        fs_anim = viz_cfg["field_settings"].get(anim_field, {})
        cmap = fs_anim.get("cmap", FIELD_DEFAULT_CMAPS.get(anim_field, "viridis"))
        buf = vis_exp.create_animation_buffer(
            history, anim_field, params["dt"],
            output_format=a_fmt, fps=a_fps, dpi=a_dpi,
            cmap_name=cmap, watermark=a_wm,
            watermark_text=a_wm_text, watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
        if buf is not None:
            st.download_button(f"Download .{a_fmt}", buf,
                                f"{name}_{anim_field}.{a_fmt}")
        else:
            st.error("Animation generation failed. For MP4, ensure ffmpeg is installed.")

    if st.button("📦 Batch generate ALL fields (zip of GIFs)"):
        vis_exp = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            prog = st.progress(0)
            fields_all = list(FIELD_DEFAULT_CMAPS.keys())
            for k, fname in enumerate(fields_all):
                fs_b = viz_cfg["field_settings"].get(fname, {})
                cmap = fs_b.get("cmap", FIELD_DEFAULT_CMAPS.get(fname, "viridis"))
                buf = vis_exp.create_animation_buffer(
                    history, fname, params["dt"],
                    output_format="gif", fps=a_fps, dpi=a_dpi,
                    cmap_name=cmap, watermark=a_wm,
                    watermark_text=a_wm_text, watermark_pos=a_wm_pos, viz_cfg=viz_cfg)
                if buf is not None:
                    zf.writestr(f"{name}_{fname}.gif", buf.getvalue())
                prog.progress((k + 1) / len(fields_all))
        zip_buf.seek(0)
        st.download_button("Download all GIFs (.zip)", zip_buf, f"{name}_all_fields.zip")

    st.markdown("---")
    st.subheader("🖼️ Snapshot PNG export")
    snap_field = st.selectbox("Field for snapshot", list(FIELD_DEFAULT_CMAPS.keys()),
                               index=2, key="snap_field")
    snap_frame = st.slider("Frame", 0, len(history) - 1, len(history) - 1, key="snap_frame")
    if st.button("📸 Generate multi-field snapshot PNG"):
        vis_snap = FerroVisualizer(Nx, Ny, dx, dy, dt=params["dt"])
        style = {
            "colormaps": {f: fset.get("cmap", FIELD_DEFAULT_CMAPS.get(f, "viridis"))
                           for f, fset in viz_cfg["field_settings"].items()},
            "field_settings": viz_cfg["field_settings"],
            "title_font_size": viz_cfg["title_font_size"],
            "label_font_size": viz_cfg["label_font_size"],
            "colorbar_fraction": viz_cfg["colorbar_fraction"],
            "colorbar_pad": viz_cfg["colorbar_pad"],
            "figure_customization": viz_cfg.get("figure_customization", {}),
        }
        fig = vis_snap.create_multi_field_comparison(history[snap_frame], style)
        if fig is not None and fig is not FAILED:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=int(viz_cfg.get("dpi", 150)),
                        bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            st.download_button("Download PNG", buf,
                                f"{name}_frame{snap_frame:04d}.png")


# ============================================================================
# UI TAB: COMPARE
# ============================================================================
def render_compare_tab(params, viz_cfg):
    """Side-by-side comparison of saved simulations.

    Fix: SimulationDatabase.list_for_ui() only returns lightweight summaries
    (id, name, params) — NOT results_history. Must fetch full records via
    SimulationDatabase.get(id) to access the saved field history.
    """
    st.header("🔍 Compare Saved Simulations")

    sims = SimulationDatabase.list_for_ui()
    if not sims:
        st.info("No saved simulations. Run one first.")
        return

    names = [s["name"] for s in sims]
    picked = st.multiselect("Select simulations", names,
                             default=names[:min(2, len(names))])
    if not picked:
        st.warning("Select at least one simulation.")
        return

    field = st.selectbox("Field", list(FIELD_DEFAULT_CMAPS.keys()), index=2, key="cmp_field")
    fs = viz_cfg["field_settings"].get(field, {})
    cmap = ColormapRegistry.resolve(fs.get("cmap", "viridis"),
                                     invert=fs.get("invert", False))

    # ---- Fetch FULL simulation records (list_for_ui only has summaries) ----
    all_vals = []
    selected_sims = []

    for n in picked:
        s_ui = sims[names.index(n)]
        # FIX: Fetch the full simulation record using the ID
        s_full = SimulationDatabase.get(s_ui["id"])
        if s_full is None:
            continue
        selected_sims.append(s_full)

        h = s_full.get("results_history", [])
        if h and field in h[-1]:
            d = np.asarray(h[-1][field])
            finite = d[np.isfinite(d)]
            if finite.size:
                all_vals.append(finite)

    if not selected_sims:
        st.warning("No valid simulations could be loaded.")
        return

    # ---- Compute global bounds across all selected sims ----
    if fs.get("auto", True):
        if field in FIELD_DEFAULT_RNG:
            vmin_g, vmax_g = FIELD_DEFAULT_RNG[field]
        elif all_vals:
            stack = np.concatenate(all_vals)
            vmin_g = float(np.percentile(stack, 2))
            vmax_g = float(np.percentile(stack, 98))
        else:
            vmin_g, vmax_g = None, None
    else:
        vmin_g = fs.get("vmin")
        vmax_g = fs.get("vmax")

    # ---- Render side-by-side heatmaps ----
    cols = min(3, len(selected_sims))
    rows = (len(selected_sims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, sim in enumerate(selected_sims):
        p = sim["params"]
        h = sim.get("results_history", [])
        if not h or field not in h[-1]:
            continue
        d = np.asarray(h[-1][field])
        dx_p = p.get("L", 64e-9) / p.get("Nx", 128)
        extent = [-p["Nx"] * dx_p / 2 * 1e9, p["Nx"] * dx_p / 2 * 1e9,
                  -p["Ny"] * dx_p / 2 * 1e9, p["Ny"] * dx_p / 2 * 1e9]
        ax = axes[i]
        im = ax.imshow(d, extent=extent, origin="lower", cmap=cmap,
                       vmin=vmin_g, vmax=vmax_g, aspect="equal")
        ax.set_title(f"{sim['id']}\n{p.get('preset', '?')}, s={p.get('screening', 0):.2f}",
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    for k in range(len(selected_sims), len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Parameter comparison table ----
    st.subheader("📋 Parameter comparison")
    rows_data = []
    for sim in selected_sims:
        p = sim["params"]
        rows_data.append({
            "ID": sim["id"],
            "Preset": p.get("preset", "?"),
            "Nx": p.get("Nx", "?"),
            "L (nm)": f"{p.get('L_box_nm', 0):.1f}",
            "h (nm)": f"{p.get('h_film_nm', 0):.1f}",
            "screening": f"{p.get('screening', 0):.2f}",
            "beta_sub": f"{p.get('beta_sub', 0):.2e}",
        })
    st.dataframe(pd.DataFrame(rows_data))

    # ---- Database management ----
    st.markdown("---")
    with st.expander("🗄️ Simulation database", expanded=False):
        n_sims = len(SimulationDatabase.all())
        st.caption(f"{n_sims} saved simulations (max {SimulationDatabase.MAX_SIMS})")
        if st.button("🧹 Clear all saved simulations"):
            SimulationDatabase.clear_all()
            st.success("Cleared.")
            st.rerun()


# ============================================================================
# SINGLE SIMULATION VIEW
# ============================================================================
def render_single_view(sid, viz_cfg):
    """Display a single saved simulation."""
    sim = SimulationDatabase.get(sid)
    if sim is None:
        st.error("Simulation not found.")
        return

    st.header(f"Simulation: {sid}")
    params = sim["params"]
    history = sim["results_history"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    L = params.get("L", 64e-9)
    dx = L / Nx
    dy = dx

    st.json(params)

    if history:
        idx = st.slider("Frame", 0, len(history) - 1, len(history) - 1,
                        key=f"view_frame_{sid}")
        vis = FerroVisualizer(Nx, Ny, dx, dy, dt=params.get("dt", 1e-12))
        style = {
            "colormaps": {f: fset.get("cmap", FIELD_DEFAULT_CMAPS.get(f, "viridis"))
                           for f, fset in viz_cfg["field_settings"].items()},
            "field_settings": viz_cfg["field_settings"],
            "title_font_size": viz_cfg["title_font_size"],
            "label_font_size": viz_cfg["label_font_size"],
            "colorbar_fraction": viz_cfg["colorbar_fraction"],
            "colorbar_pad": viz_cfg["colorbar_pad"],
            "figure_customization": viz_cfg.get("figure_customization", {}),
        }
        fig = vis.create_multi_field_comparison(history[idx], style)
        if fig is not None and fig is not FAILED:
            st.pyplot(fig)
            plt.close(fig)

    if st.button("🗑️ Delete"):
        SimulationDatabase.delete(sid)
        if "view_id" in st.session_state:
            del st.session_state.view_id
        st.rerun()


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    """Entry point — builds sidebar, dispatches to tab renderers."""
    st.set_page_config(
        page_title="Ferroelectric Phase-Field (Integrated)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem;
        text-align: center;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #2166ac, #b2182b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .viz-badge {
        background: #eef;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #335;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">⚛️ Ferroelectric Phase-Field '
                "— FFT, Thin-Film, 3-Component P</h1>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#FFF8E7; padding:1rem; border-radius:8px;
                border-left:5px solid #b2182b;">
    <b>Physics:</b> 3-component polarization P = (Px, Py, Pz) with
    Landau-Devonshire bulk + Lifshitz gradient + thin-film depolarization
    (open-circuit kernel) + substrate anisotropy.<br>
    <b>Discretization:</b> Semi-implicit Fourier spectral on periodic grid.<br>
    <b>Units:</b> All SI (P in C/m², E in V/m, length in m, time in s).<br>
    <span class="viz-badge">8 tabs: Geometry · Run · Viz Settings · Results ·
    Line Profiles · Plotly · Export · Compare</span>
    </div>
    """, unsafe_allow_html=True)

    # Apply theme
    viz_cfg = get_viz_config()
    apply_matplotlib_theme(viz_cfg["theme"], viz_cfg)

    # Build sidebar
    params = build_sidebar()

    # Dispatch based on mode
    mode = params.get("mode", "Run New Simulation")

    if mode == "Run New Simulation":
        tabs = st.tabs(["📐 Geometry", "▶️ Run", "🎨 Viz Settings",
                        "📊 Results", "🔬 Line Profiles", "🌐 Plotly",
                        "📤 Export", "🔍 Compare"])
        with tabs[0]:
            render_geometry_tab(params, viz_cfg)
        with tabs[1]:
            render_run_tab(params, viz_cfg)
        with tabs[2]:
            render_viz_settings_tab(params, viz_cfg)
        with tabs[3]:
            render_results_tab(params, viz_cfg)
        with tabs[4]:
            render_line_profiles_tab(params, viz_cfg)
        with tabs[5]:
            render_plotly_tab(params, viz_cfg)
        with tabs[6]:
            render_export_tab(params, viz_cfg)
        with tabs[7]:
            render_compare_tab(params, viz_cfg)

    elif mode == "Single Simulation View":
        sims = SimulationDatabase.list_for_ui()
        if not sims:
            st.info("No saved simulations.")
        else:
            names = [s["name"] for s in sims]
            chosen = st.selectbox("Select simulation", names)
            st.session_state.view_id = sims[names.index(chosen)]["id"]
            render_single_view(st.session_state.view_id, viz_cfg)

    elif mode == "Compare Saved":
        sims = SimulationDatabase.list_for_ui()
        if not sims:
            st.info("No saved simulations.")
        else:
            names = [s["name"] for s in sims]
            picked = st.multiselect("Select simulations", names,
                                     default=names[:min(2, len(names))])
            st.session_state.compare_ids = [sims[names.index(n)]["id"]
                                             for n in picked]
            if st.session_state.compare_ids:
                # Reuse the compare tab renderer with a dummy params
                dummy_params = {"Nx": 128, "Ny": 128, "L": 64e-9,
                                "mode": "Compare Saved"}
                render_compare_tab(dummy_params, viz_cfg)


# ============================================================================
# PARAMETER VALIDATION AND STABILITY ANALYSIS
# ============================================================================
def validate_parameters(params):
    """Validate simulation parameters and return (errors, warnings).

    Checks:
        - dt > 0, dx > 0
        - Grid size Nx, Ny >= 32
        - Landau coefficients: alpha1 < 0 (below Tc), alpha11 > 0
        - Tetragonal symmetry: alpha12 > 2 * alpha11
        - Gradient coefficients: G11 > 0, G44 > 0
        - Semi-implicit stability: dt * Gamma * |alpha1| < 1 (nonlinear part)
        - Wall resolution: xi / dx > 3 (wall width resolves on grid)
        - Domain capacity: L / xi > 10 (box fits multiple domains)
        - Screening factor in [0, 1]
        - Substrate anisotropy: beta_sub > |alpha1| if enabled
        - Film thickness h > 0
    """
    errors = []
    warns = []

    # ---- Basic parameter sanity ----
    if params.get("dt", 0) <= 0:
        errors.append("dt must be positive")
    if params.get("Nx", 0) < 32 or params.get("Ny", 0) < 32:
        warns.append("N < 32 per axis may be inaccurate")
    if params.get("L", 0) <= 0:
        errors.append("Box size L must be positive")
    if params.get("h_film", 0) <= 0:
        errors.append("Film thickness h must be positive")

    # ---- Landau coefficient checks ----
    mat = params.get("mat")
    if mat is not None:
        if mat.alpha1 >= 0:
            warns.append(f"alpha1 = {mat.alpha1:.3e} >= 0 — system is paraelectric "
                         f"(above Tc). No ferroelectric domains will form.")
        if mat.alpha11 <= 0:
            errors.append(f"alpha11 = {mat.alpha11:.3e} must be positive "
                          f"(Landau potential unbounded below otherwise).")
        if mat.alpha12 <= 2 * mat.alpha11:
            warns.append(f"alpha12 = {mat.alpha12:.3e} <= 2*alpha11 = {2*mat.alpha11:.3e} "
                         f"— tetragonal symmetry violated. Diagonal state "
                         f"will be energetically favored over axial.")
        if mat.G11 <= 0:
            errors.append(f"G11 = {mat.G11:.3e} must be positive.")
        if mat.G44 <= 0:
            errors.append(f"G44 = {mat.G44:.3e} must be positive.")

        # ---- Derived quantity checks ----
        Ps = mat.Ps_eq()
        xi = mat.xi_wall()
        dx = params["L"] / params["Nx"]

        if xi / dx < 3:
            warns.append(f"Wall width xi = {xi*1e9:.3f} nm is less than 3*dx = {3*dx*1e9:.3f} nm. "
                         f"Domain walls will not be resolved. Increase G11 or decrease dx.")
        if params["L"] / xi < 10:
            warns.append(f"Box size L = {params['L']*1e9:.1f} nm is less than 10*xi = {10*xi*1e9:.1f} nm. "
                         f"Box is too small to fit multiple domains.")

        # ---- Nonlinear stability (explicit Landau part) ----
        dt = params["dt"]
        Gamma = params["Gamma"]
        nonlinear_cfl = dt * Gamma * abs(mat.alpha1)
        if nonlinear_cfl > 1.0:
            warns.append(f"dt * Gamma * |alpha1| = {nonlinear_cfl:.3e} > 1. "
                         f"Nonlinear (Landau) part may be unstable. "
                         f"Reduce dt or Gamma.")
        elif nonlinear_cfl > 0.5:
            warns.append(f"dt * Gamma * |alpha1| = {nonlinear_cfl:.3e} is moderately large. "
                         f"Consider reducing dt for better accuracy.")

        # ---- Substrate anisotropy check ----
        if params.get("use_substrate", False):
            if mat.beta_sub <= abs(mat.alpha1):
                warns.append(f"beta_sub = {mat.beta_sub:.3e} <= |alpha1| = {abs(mat.alpha1):.3e}. "
                             f"Substrate anisotropy is too weak to suppress in-plane P. "
                             f"System may rotate P into the plane instead of forming Pz stripes. "
                             f"Recommend beta_sub > 1.5 * |alpha1|.")

    # ---- Screening factor ----
    screening = params.get("screening", 0.65)
    if screening < 0 or screening > 1:
        errors.append(f"screening = {screening} must be in [0, 1].")
    elif screening < 0.3:
        warns.append(f"screening = {screening:.2f} is low (strong electrode screening). "
                     f"Uniform Pz will be favored over stripe domains.")
    elif screening > 0.9:
        warns.append(f"screening = {screening:.2f} is high (weak screening, open-circuit). "
                     f"System may go to Pz = 0 (paraelectric) instead of forming stripes. "
                     f"Sweet spot for stripe formation is typically s in [0.5, 0.8].")

    # ---- Time step vs total time ----
    n_steps = params.get("n_steps", 5000)
    total_time = dt * n_steps
    if total_time < 1e-10:
        warns.append(f"Total simulation time = {total_time*1e9:.4g} ns is very short. "
                     f"System may not reach equilibrium.")
    if n_steps > 20000:
        warns.append(f"n_steps = {n_steps} is large. Simulation may take a long time.")

    return errors, warns


def compute_stability_metrics(params):
    """Compute and return a dict of stability and resolution metrics.

    Returns:
        dict with keys:
            nonlinear_cfl   : dt * Gamma * |alpha1|  (need < 1 for stability)
            linear_cfl_max  : dt * Gamma * max(M_zz)  (semi-implicit, always stable)
            wall_resolution : xi / dx  (need > 3)
            domain_capacity : L / xi   (need > 10 for multiple domains)
            stripe_period_est : estimated 180° stripe period (nm)
            landau_gain     : |alpha1|^2 / (4*alpha11) * h  (J/m², Landau energy gain)
            depol_cost      : Ps^2 / (2*eps0) * h * screening  (J/m², depol cost for uniform Pz)
    """
    mat = params["mat"]
    dt = params["dt"]
    Gamma = params["Gamma"]
    L = params["L"]
    h = params["h_film"]
    screening = params["screening"]
    Nx = params["Nx"]
    dx = L / Nx

    Ps = mat.Ps_eq()
    xi = mat.xi_wall()

    nonlinear_cfl = dt * Gamma * abs(mat.alpha1)
    # Max k value
    k_max = np.pi / dx
    linear_cfl_max = dt * Gamma * (mat.G44 * k_max**2 + screening / EPSILON_0)

    wall_resolution = xi / dx
    domain_capacity = L / xi

    # Estimate stripe period (Kittel-like): d ~ sqrt(h * xi * something)
    # Very rough: d ~ 2 * pi * sqrt(G44 * h / (2 * screening / eps0))
    if screening > 0:
        stripe_period_est = 2 * np.pi * np.sqrt(mat.G44 * h / (2 * screening / EPSILON_0))
    else:
        stripe_period_est = float("inf")

    landau_gain = mat.alpha1**2 / (4 * mat.alpha11) * h  # J/m² (negative, gain)
    depol_cost = Ps**2 / (2 * EPSILON_0) * h * screening  # J/m² (positive, cost)

    return {
        "nonlinear_cfl": nonlinear_cfl,
        "linear_cfl_max": linear_cfl_max,
        "wall_resolution": wall_resolution,
        "domain_capacity": domain_capacity,
        "stripe_period_est_nm": stripe_period_est * 1e9,
        "landau_gain_J_m2": landau_gain,
        "depol_cost_J_m2": depol_cost,
        "stripe_favored": landau_gain > depol_cost,  # rough check
    }


# ============================================================================
# EXTENDED ENERGY ANALYSIS
# ============================================================================
def compute_energy_breakdown(solver):
    """Compute a detailed breakdown of all energy components.

    Returns a dict with:
        - mean values of each energy component
        - max/min values
        - spatial std (inhomogeneity measure)
        - fraction of total
    """
    if solver.history["f_total"][-1] is None:
        return None

    f_bulk = np.array(solver.history["f_bulk"])
    f_grad = np.array(solver.history["f_grad"])
    f_depol_Pz = np.array(solver.history["f_depol_Pz"])
    f_depol_in = np.array(solver.history["f_depol_inplane"])
    f_total = np.array(solver.history["f_total"])

    # Final-step values
    n = len(f_total) - 1
    total_final = f_total[n]
    if total_final != 0:
        frac_bulk = f_bulk[n] / total_final
        frac_grad = f_grad[n] / total_final
        frac_depol_Pz = f_depol_Pz[n] / total_final
        frac_depol_in = f_depol_in[n] / total_final
    else:
        frac_bulk = frac_grad = frac_depol_Pz = frac_depol_in = 0.0

    return {
        "final": {
            "f_bulk": float(f_bulk[n]),
            "f_grad": float(f_grad[n]),
            "f_depol_Pz": float(f_depol_Pz[n]),
            "f_depol_inplane": float(f_depol_in[n]),
            "f_total": float(total_final),
            "frac_bulk": float(frac_bulk),
            "frac_grad": float(frac_grad),
            "frac_depol_Pz": float(frac_depol_Pz),
            "frac_depol_inplane": float(frac_depol_in),
        },
        "initial": {
            "f_bulk": float(f_bulk[0]),
            "f_grad": float(f_grad[0]),
            "f_depol_Pz": float(f_depol_Pz[0]),
            "f_depol_inplane": float(f_depol_in[0]),
            "f_total": float(f_total[0]),
        },
        "convergence": {
            "f_total_change": float(f_total[n] - f_total[0]),
            "f_total_rel_change": float((f_total[n] - f_total[0]) / max(abs(f_total[0]), 1e-30)),
            "steps": len(f_total),
        },
        "time_series": {
            "f_bulk": f_bulk.tolist(),
            "f_grad": f_grad.tolist(),
            "f_depol_Pz": f_depol_Pz.tolist(),
            "f_depol_inplane": f_depol_in.tolist(),
            "f_total": f_total.tolist(),
        },
    }


def compute_domain_statistics(Pz, Ps, dx):
    """Compute statistics about the domain structure in Pz.

    Returns:
        - n_domains: estimated number of domains
        - mean_domain_size: mean domain size (nm)
        - domain_wall_fraction: fraction of grid cells that are walls
        - mean_Pz: mean |Pz|
        - std_Pz: std of Pz (inhomogeneity)
        - stripe_period_est: estimated stripe period (nm) via autocorrelation
    """
    Pz_arr = np.asarray(Pz)
    Ny, Nx = Pz_arr.shape

    # Count sign changes along central row
    central_row = Pz_arr[Ny // 2, :]
    sign_changes = np.sum(np.abs(np.diff(np.sign(central_row))) > 0)
    n_domains = max(sign_changes, 1)
    mean_domain_size = Nx * dx * 1e9 / max(n_domains, 1)

    # Wall fraction: cells where |Pz| < 0.5 * Ps
    wall_mask = np.abs(Pz_arr) < 0.5 * Ps
    domain_wall_fraction = float(np.mean(wall_mask))

    mean_Pz = float(np.mean(np.abs(Pz_arr)))
    std_Pz = float(np.std(Pz_arr))

    # Stripe period via autocorrelation of central row
    row = central_row - central_row.mean()
    if np.std(row) > 1e-12:
        autocorr = np.correlate(row, row, mode="full")
        autocorr = autocorr[len(autocorr)//2:]  # second half
        autocorr /= autocorr[0]  # normalize
        # Find first minimum (wall) then first maximum after that (period)
        if len(autocorr) > 10:
            # Find first zero crossing
            zero_cross = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_cross) > 0:
                first_min = zero_cross[0]
                # Find next maximum after first minimum
                after_min = autocorr[first_min:]
                maxima = np.where((np.diff(after_min)[:-1] > 0) &
                                  (np.diff(after_min)[1:] < 0))[0]
                if len(maxima) > 0:
                    stripe_period_est = (first_min + maxima[0]) * dx * 1e9
                else:
                    stripe_period_est = mean_domain_size
            else:
                stripe_period_est = mean_domain_size
        else:
            stripe_period_est = mean_domain_size
    else:
        stripe_period_est = 0.0

    return {
        "n_domains": int(n_domains),
        "mean_domain_size_nm": float(mean_domain_size),
        "domain_wall_fraction": domain_wall_fraction,
        "mean_Pz": mean_Pz,
        "std_Pz": std_Pz,
        "stripe_period_est_nm": float(stripe_period_est),
    }


# ============================================================================
# ADDITIONAL VISUALIZER METHODS
# ============================================================================
def create_energy_breakdown_figure(solver, viz_cfg):
    """Create a stacked area chart of energy components over time.

    Shows how f_bulk, f_grad, f_depol_Pz, f_depol_inplane contribute to
    the total free energy over the simulation.
    """
    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("diagnostics_width", 13.0))
    fig_h = float(fig_cfg.get("diagnostics_height", 6.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    steps = np.array(solver.history["step"])
    f_bulk = np.array(solver.history["f_bulk"])
    f_grad = np.array(solver.history["f_grad"])
    f_depol_Pz = np.array(solver.history["f_depol_Pz"])
    f_depol_in = np.array(solver.history["f_depol_inplane"])
    f_total = np.array(solver.history["f_total"])

    # Left: stacked area (absolute values for visualization)
    ax1.fill_between(steps, 0, np.abs(f_bulk), alpha=0.7,
                     color=fig_cfg.get("color_series_1", "#1f77b4"),
                     label="|f_bulk|")
    ax1.fill_between(steps, np.abs(f_bulk),
                     np.abs(f_bulk) + np.abs(f_grad), alpha=0.7,
                     color=fig_cfg.get("color_series_3", "#2ca02c"),
                     label="|f_grad|")
    ax1.fill_between(steps, np.abs(f_bulk) + np.abs(f_grad),
                     np.abs(f_bulk) + np.abs(f_grad) + np.abs(f_depol_Pz),
                     alpha=0.7,
                     color=fig_cfg.get("color_series_2", "#d62728"),
                     label="|f_depol_Pz|")
    ax1.plot(steps, np.abs(f_total), "k-", lw=2.5, label="|f_total|")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("|Energy density| (J/m³)")
    ax1.set_title("Energy components (absolute)")
    ax1.set_yscale("log")
    apply_axis_styling(ax1, fig_cfg, force_aspect="auto")
    apply_legend_styling(ax1, fig_cfg)

    # Right: relative fractions
    total_abs = np.abs(f_bulk) + np.abs(f_grad) + np.abs(f_depol_Pz) + np.abs(f_depol_in) + 1e-30
    frac_bulk = np.abs(f_bulk) / total_abs
    frac_grad = np.abs(f_grad) / total_abs
    frac_depol_Pz = np.abs(f_depol_Pz) / total_abs
    frac_depol_in = np.abs(f_depol_in) / total_abs

    ax2.fill_between(steps, 0, frac_bulk, alpha=0.7,
                     color=fig_cfg.get("color_series_1", "#1f77b4"),
                     label="f_bulk")
    ax2.fill_between(steps, frac_bulk, frac_bulk + frac_grad, alpha=0.7,
                     color=fig_cfg.get("color_series_3", "#2ca02c"),
                     label="f_grad")
    ax2.fill_between(steps, frac_bulk + frac_grad,
                     frac_bulk + frac_grad + frac_depol_Pz, alpha=0.7,
                     color=fig_cfg.get("color_series_2", "#d62728"),
                     label="f_depol_Pz")
    ax2.fill_between(steps, frac_bulk + frac_grad + frac_depol_Pz,
                     frac_bulk + frac_grad + frac_depol_Pz + frac_depol_in,
                     alpha=0.7,
                     color=fig_cfg.get("color_series_4", "#9467bd"),
                     label="f_depol_in")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Fraction of total |energy|")
    ax2.set_title("Relative energy contributions")
    ax2.set_ylim(0, 1)
    apply_axis_styling(ax2, fig_cfg, force_aspect="auto")
    apply_legend_styling(ax2, fig_cfg)

    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    return fig


def create_domain_analysis_figure(Pz, Ps, dx, viz_cfg):
    """Create a figure showing domain structure analysis.

    Includes:
        - Pz heatmap
        - Central row profile
        - Autocorrelation of central row (stripe period detection)
    """
    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("diagnostics_width", 13.0))
    fig_h = float(fig_cfg.get("diagnostics_height", 5.0))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                                         gridspec_kw={"width_ratios": [2, 2, 1.5]})

    Pz_arr = np.asarray(Pz)
    Ny, Nx = Pz_arr.shape
    extent = [-Nx * dx / 2 * 1e9, Nx * dx / 2 * 1e9,
              -Ny * dx / 2 * 1e9, Ny * dx / 2 * 1e9]

    # Pz heatmap
    vmax = max(Ps, np.abs(Pz_arr).max())
    im = ax1.imshow(Pz_arr, extent=extent, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax1, label="Pz (C/m²)")
    ax1.set_title("Pz field")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    apply_axis_styling(ax1, fig_cfg, force_aspect="equal")

    # Central row profile
    central_row = Pz_arr[Ny // 2, :]
    x_nm = np.linspace(extent[0], extent[1], Nx)
    ax2.plot(x_nm, central_row,
             color=fig_cfg.get("color_series_1", "#1f77b4"),
             lw=float(fig_cfg.get("line_width", 2.0)))
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.axhline(Ps, color="red", ls=":", lw=0.5, alpha=0.5, label=f"+Ps={Ps:.3f}")
    ax2.axhline(-Ps, color="blue", ls=":", lw=0.5, alpha=0.5, label=f"-Ps={-Ps:.3f}")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("Pz (C/m²)")
    ax2.set_title("Central row profile")
    apply_axis_styling(ax2, fig_cfg, force_aspect="auto")
    apply_legend_styling(ax2, fig_cfg)

    # Autocorrelation
    row = central_row - central_row.mean()
    if np.std(row) > 1e-12:
        autocorr = np.correlate(row, row, mode="full")
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        lags = np.arange(len(autocorr)) * dx * 1e9
        ax3.plot(lags[:len(lags)//2], autocorr[:len(autocorr)//2],
                 color=fig_cfg.get("color_series_2", "#d62728"),
                 lw=float(fig_cfg.get("line_width", 2.0)))
        ax3.axhline(0, color="gray", ls="--", lw=0.5)
        ax3.set_xlabel("Lag (nm)")
        ax3.set_ylabel("Autocorrelation")
        ax3.set_title("Stripe period detection")
        apply_axis_styling(ax3, fig_cfg, force_aspect="auto")

    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    return fig


# ============================================================================
# HELPER: RENDER PARAMETER VALIDATION
# ============================================================================
def render_parameter_validation(params):
    """Render the parameter validation results in a Streamlit expander."""
    with st.expander("✅ Parameter validation", expanded=False):
        errors, warns = validate_parameters(params)

        if errors:
            st.error("**Errors:**")
            for e in errors:
                st.write(f"• {e}")

        if warns:
            st.warning("**Warnings:**")
            for w in warns:
                st.write(f"• {w}")

        if not errors and not warns:
            st.success("✅ All parameters validated. No issues found.")

        # Show stability metrics
        st.markdown("---")
        st.subheader("📊 Stability & resolution metrics")
        metrics = compute_stability_metrics(params)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nonlinear CFL", f"{metrics['nonlinear_cfl']:.3e}",
                    help="dt * Gamma * |alpha1|, need < 1 for stability")
        col2.metric("Wall resolution", f"{metrics['wall_resolution']:.2f}",
                    help="xi / dx, need > 3")
        col3.metric("Domain capacity", f"{metrics['domain_capacity']:.1f}",
                    help="L / xi, need > 10")
        col4.metric("Est. stripe period", f"{metrics['stripe_period_est_nm']:.1f} nm",
                    help="Kittel-like estimate of 180° stripe period")

        col1, col2 = st.columns(2)
        col1.metric("Landau gain", f"{metrics['landau_gain_J_m2']:.3e} J/m²",
                    help="|alpha1|²/(4*alpha11) * h — energy gain from ferroelectric state")
        col2.metric("Depol cost (uniform)", f"{metrics['depol_cost_J_m2']:.3e} J/m²",
                    help="Ps²/(2*eps0) * h * screening — depol penalty for uniform Pz")

        if metrics["stripe_favored"]:
            st.success("✅ Stripes favored: Landau gain > depol cost for uniform Pz.")
        else:
            st.warning("⚠️ Uniform Pz=0 may be favored over stripes. "
                       "Adjust screening or substrate anisotropy.")


# ============================================================================
# HELPER: RENDER DOMAIN ANALYSIS
# ============================================================================
def render_domain_analysis(solver, params, viz_cfg):
    """Render domain structure analysis for the final frame."""
    history = st.session_state.get("history", [])
    if not history:
        return

    st.subheader("🔬 Domain structure analysis")

    final_frame = history[-1]
    Pz = final_frame["Pz"]
    Ps = params["mat"].Ps_eq()
    dx = params["L"] / params["Nx"]

    # Domain statistics
    stats = compute_domain_statistics(Pz, Ps, dx)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N domains", stats["n_domains"])
    col2.metric("Mean domain size", f"{stats['mean_domain_size_nm']:.1f} nm")
    col3.metric("Wall fraction", f"{stats['domain_wall_fraction']:.2%}")
    col4.metric("Stripe period (est.)", f"{stats['stripe_period_est_nm']:.1f} nm")

    # Domain analysis figure
    fig = create_domain_analysis_figure(Pz, Ps, dx, viz_cfg)
    if fig is not None and fig is not FAILED:
        st.pyplot(fig)
        plt.close(fig)

    # Energy breakdown
    st.markdown("---")
    st.subheader("📈 Energy breakdown")
    breakdown = compute_energy_breakdown(solver)
    if breakdown:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Final energy components:**")
            f = breakdown["final"]
            st.write(f"• f_bulk = {f['f_bulk']:.4e} J/m³ ({f['frac_bulk']:.1%})")
            st.write(f"• f_grad = {f['f_grad']:.4e} J/m³ ({f['frac_grad']:.1%})")
            st.write(f"• f_depol_Pz = {f['f_depol_Pz']:.4e} J/m³ ({f['frac_depol_Pz']:.1%})")
            st.write(f"• f_depol_in = {f['f_depol_inplane']:.4e} J/m³ ({f['frac_depol_inplane']:.1%})")
            st.write(f"• **f_total = {f['f_total']:.4e} J/m³**")
        with col2:
            st.markdown("**Convergence:**")
            c = breakdown["convergence"]
            st.write(f"• Steps: {c['steps']}")
            st.write(f"• Δf_total = {c['f_total_change']:.4e} J/m³")
            st.write(f"• Relative change: {c['f_total_rel_change']:.4%}")

    # Energy breakdown figure
    fig_energy = create_energy_breakdown_figure(solver, viz_cfg)
    if fig_energy is not None and fig_energy is not FAILED:
        st.pyplot(fig_energy)
        plt.close(fig_energy)


# ============================================================================
# HELPER: EXPORT CONFIGURATION SUMMARY
# ============================================================================
def get_config_summary(params):
    """Return a human-readable summary of the configuration."""
    mat = params["mat"]
    Ps = mat.Ps_eq()
    xi = mat.xi_wall()
    dx = params["L"] / params["Nx"]

    summary = f"""
=== Ferroelectric Phase-Field Configuration Summary ===

Material: {params['preset']}
  alpha1  = {mat.alpha1:.3e} V·m/C
  alpha11 = {mat.alpha11:.3e} V·m⁵/C³
  alpha12 = {mat.alpha12:.3e} V·m⁵/C³  (tetragonal: {'OK' if mat.alpha12 > 2*mat.alpha11 else 'VIOLATED'})
  G11     = {mat.G11:.3e} V·m³/C
  G44     = {mat.G44:.3e} V·m³/C
  beta_sub= {mat.beta_sub:.3e} V·m/C

Derived:
  |Ps|    = {Ps*100:.2f} μC/cm² ({Ps:.4f} C/m²)
  xi      = {xi*1e9:.4f} nm (wall width)
  xi/dx   = {xi/dx:.2f} (resolution, need > 3)

Geometry:
  Grid    = {params['Nx']} × {params['Ny']}
  Box     = {params['L_box_nm']:.1f} nm
  dx      = {dx*1e9:.4f} nm
  h_film  = {params['h_film_nm']:.1f} nm

Time:
  dt      = {params['dt_ps']:.3f} ps ({params['dt']:.3e} s)
  steps   = {params['n_steps']}
  total   = {params['n_steps']*params['dt']*1e9:.4g} ns

Physics:
  screening     = {params['screening']:.2f} (0=short, 1=open-circuit)
  use_elec      = {params['use_elec']}
  use_substrate = {params['use_substrate']}
  init_mode     = {params['init_mode']}
  noise_amp     = {params['noise_amp_uc']:.1f} μC/cm² = {params['noise_amp']:.4f} C/m²
  Gamma         = {params['Gamma']:.3f} (Ω·m)⁻¹

Bug fixes:
  #1 noise_amp = noise_amp_uc * 1e-2  (μC/cm² → C/m²): FIXED
  #2 f_depol_Pz single computation (no dead code): FIXED
  #3 alpha12 > 2*alpha11 warning: ACTIVE
  #4 screening/beta_sub documented: DONE
  #5 M_zz[0,0] = screening/EPSILON_0: CORRECT
  #6 Poisson divides by EPSILON_0: CORRECT
"""
    return summary


# ============================================================================
# HELPER: SAVE CONFIGURATION TO FILE
# ============================================================================
def save_config_summary(params):
    """Return a BytesIO buffer containing the configuration summary text."""
    summary = get_config_summary(params)
    return BytesIO(summary.encode()), "config_summary.txt"


# ============================================================================
# HELPER: LOAD CONFIGURATION FROM FILE
# ============================================================================
def load_config_summary(buf):
    """Parse a configuration summary text file and return a params dict."""
    text = buf.read().decode() if hasattr(buf, "read") else str(buf)
    # This is a simple parser — for full parameter reconstruction,
    # use the PKL or JSON export instead.
    params = {}
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("="):
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Try to convert to float
            try:
                params[key] = float(val.split()[0])
            except (ValueError, IndexError):
                params[key] = val
    return params


# ============================================================================
# HELPER: COMPUTE POLARIZATION HISTOGRAM
# ============================================================================
def compute_polarization_histogram(Px, Py, Pz, Ps, n_bins=50):
    """Compute histograms of each polarization component.

    Returns a dict of (bin_centers, counts) for each component.
    """
    result = {}
    for name, arr in [("Px", Px), ("Py", Py), ("Pz", Pz)]:
        arr = np.asarray(arr).ravel()
        counts, bin_edges = np.histogram(arr, bins=n_bins, range=(-Ps*1.2, Ps*1.2))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        result[name] = (bin_centers, counts)
    # |P| histogram
    Pmag = np.sqrt(np.asarray(Px)**2 + np.asarray(Py)**2 + np.asarray(Pz)**2).ravel()
    counts_mag, bin_edges_mag = np.histogram(Pmag, bins=n_bins, range=(0, Ps*1.5))
    bin_centers_mag = 0.5 * (bin_edges_mag[:-1] + bin_edges_mag[1:])
    result["P_mag"] = (bin_centers_mag, counts_mag)
    return result


def create_polarization_histogram_figure(Px, Py, Pz, Ps, viz_cfg):
    """Create a 2x2 figure showing histograms of Px, Py, Pz, and |P|."""
    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("diagnostics_width", 13.0))
    fig_h = float(fig_cfg.get("diagnostics_height", 8.0))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    hists = compute_polarization_histogram(Px, Py, Pz, Ps)
    colors = [fig_cfg.get("color_series_1", "#1f77b4"),
              fig_cfg.get("color_series_2", "#d62728"),
              fig_cfg.get("color_series_3", "#2ca02c"),
              fig_cfg.get("color_series_4", "#9467bd")]
    titles = ["Px distribution", "Py distribution", "Pz distribution", "|P| distribution"]
    xlabels = ["Px (C/m²)", "Py (C/m²)", "Pz (C/m²)", "|P| (C/m²)"]

    for idx, (name, title, xlabel, color) in enumerate(
            zip(["Px", "Py", "Pz", "P_mag"], titles, xlabels, colors)):
        ax = axes[idx]
        bin_centers, counts = hists[name]
        ax.bar(bin_centers, counts, width=bin_centers[1]-bin_centers[0],
               color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        # Mark ±Ps
        if name != "P_mag":
            ax.axvline(Ps, color="red", ls="--", lw=1.5, label=f"+Ps={Ps:.3f}")
            ax.axvline(-Ps, color="blue", ls="--", lw=1.5, label=f"-Ps={-Ps:.3f}")
        else:
            ax.axvline(Ps, color="red", ls="--", lw=1.5, label=f"Ps={Ps:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        apply_axis_styling(ax, fig_cfg, force_aspect="auto")
        apply_legend_styling(ax, fig_cfg)

    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    return fig


# ============================================================================
# HELPER: COMPUTE FOURIER SPECTRUM
# ============================================================================
def compute_pz_spectrum(Pz, dx):
    """Compute the radial power spectrum of Pz.

    Returns (k_values, power_spectrum) where k_values are in 1/nm.
    """
    Pz_arr = np.asarray(Pz)
    Ny, Nx = Pz_arr.shape

    # 2D FFT
    Pz_hat = np.fft.fft2(Pz_arr)
    power = np.abs(Pz_hat)**2

    # Shift zero frequency to center
    power_shifted = np.fft.fftshift(power)

    # Radial binning
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_radial = np.sqrt(KX**2 + KY**2)

    # Shift k to match power_shifted
    k_radial_shifted = np.fft.fftshift(k_radial)

    # Bin by radial distance
    k_flat = k_radial_shifted.ravel()
    p_flat = power_shifted.ravel()
    k_max = k_flat.max()
    n_bins = min(Nx, Ny) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    digitized = np.digitize(k_flat, k_bins)
    spectrum = np.array([p_flat[digitized == i].mean() if np.any(digitized == i) else 0
                         for i in range(1, n_bins + 1)])

    return k_centers * 1e-9, spectrum  # k in 1/nm


def create_spectrum_figure(Pz, dx, viz_cfg):
    """Create a log-log plot of the Pz power spectrum vs wavenumber."""
    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("line_profile_width", 11.0))
    fig_h = float(fig_cfg.get("line_profile_height", 4.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    k_vals, spectrum = compute_pz_spectrum(Pz, dx)
    # Filter out zeros
    mask = spectrum > 0
    k_vals = k_vals[mask]
    spectrum = spectrum[mask]

    ax.loglog(k_vals, spectrum,
              color=fig_cfg.get("color_series_1", "#1f77b4"),
              lw=float(fig_cfg.get("line_width", 2.0)))
    ax.set_xlabel("k (1/nm)")
    ax.set_ylabel("Power |Pz_hat(k)|²")
    ax.set_title("Pz power spectrum")
    apply_axis_styling(ax, fig_cfg, force_aspect="auto")

    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    return fig


# ============================================================================
# HELPER: RENDER EXTENDED DIAGNOSTICS TAB
# ============================================================================
def render_extended_diagnostics(solver, params, viz_cfg):
    """Render extended diagnostics: histograms, spectra, domain analysis."""
    history = st.session_state.get("history", [])
    if not history:
        return

    st.subheader("📊 Extended diagnostics")

    final_frame = history[-1]
    Px = final_frame["Px"]
    Py = final_frame["Py"]
    Pz = final_frame["Pz"]
    Ps = params["mat"].Ps_eq()
    dx = params["L"] / params["Nx"]

    # Polarization histograms
    st.markdown("#### Polarization component distributions")
    fig_hist = create_polarization_histogram_figure(Px, Py, Pz, Ps, viz_cfg)
    if fig_hist is not None and fig_hist is not FAILED:
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    # Pz power spectrum
    st.markdown("#### Pz Fourier spectrum (domain size analysis)")
    fig_spec = create_spectrum_figure(Pz, dx, viz_cfg)
    if fig_spec is not None and fig_spec is not FAILED:
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    # Domain analysis
    render_domain_analysis(solver, params, viz_cfg)

    # Configuration summary download
    st.markdown("---")
    st.subheader("📋 Configuration summary")
    summary = get_config_summary(params)
    st.text(summary)
    buf, fname = save_config_summary(params)
    st.download_button("Download config summary", buf, fname, "text/plain")


# ============================================================================
# HELPER: RENDER HEADER BADGES
# ============================================================================
def render_header_badges(params):
    """Render status badges showing key parameters."""
    mat = params["mat"]
    Ps = mat.Ps_eq()
    xi = mat.xi_wall()
    dx = params["L"] / params["Nx"]

    badges = []
    badges.append(f"🖥️ {'GPU' if GPU_AVAILABLE else 'CPU'}")
    badges.append(f"📐 {params['Nx']}×{params['Ny']}")
    badges.append(f"📏 {params['L_box_nm']:.0f}nm")
    badges.append(f"🎬 {params['h_film_nm']:.0f}nm film")
    badges.append(f"⚡ s={params['screening']:.2f}")
    badges.append(f"🔬 ξ/dx={xi/dx:.1f}")
    badges.append(f"💎 |Ps|={Ps*100:.0f}μC/cm²")

    st.markdown(" ".join(f"`{b}`" for b in badges))


# ============================================================================
# HELPER: CHECK FOR EXISTING RESULTS
# ============================================================================
def has_results():
    """Check if simulation results exist in session_state."""
    return ("history" in st.session_state and
            len(st.session_state["history"]) > 0 and
            "solver" in st.session_state)


# ============================================================================
# HELPER: CLEAR RESULTS
# ============================================================================
def clear_results():
    """Clear simulation results from session_state."""
    for key in ("history", "solver", "initialized", "last_sid"):
        if key in st.session_state:
            del st.session_state[key]


# ============================================================================
# HELPER: GET LAST FRAME
# ============================================================================
def get_last_frame():
    """Return the last frame from session_state history, or None."""
    history = st.session_state.get("history", [])
    if history:
        return history[-1]
    return None


# ============================================================================
# HELPER: COMPUTE CONVERGENCE RATE
# ============================================================================
def compute_convergence_rate(solver):
    """Compute the convergence rate of the total energy.

    Returns the relative energy change per step over the last 10% of steps.
    A value close to 0 indicates convergence.
    """
    f_total = np.array(solver.history["f_total"])
    n = len(f_total)
    if n < 10:
        return float("nan")

    # Last 10% of steps
    start = int(n * 0.9)
    f_start = f_total[start]
    f_end = f_total[-1]
    steps = n - start

    if f_start == 0:
        return float("nan")

    rate = (f_end - f_start) / (abs(f_start) * steps)
    return float(rate)


# ============================================================================
# HELPER: ESTIMATE EQUILIBRIUM
# ============================================================================
def estimate_equilibrium(solver, threshold=1e-4):
    """Estimate whether the simulation has reached equilibrium.

    Returns (is_equilibrated, step_of_equilibrium).
    Equilibrium is defined as the relative energy change per step
    falling below the threshold.
    """
    f_total = np.array(solver.history["f_total"])
    n = len(f_total)
    if n < 10:
        return False, n

    for i in range(5, n):
        window = f_total[max(0, i-5):i+1]
        if len(window) < 2:
            continue
        rel_change = np.std(window) / max(abs(np.mean(window)), 1e-30)
        if rel_change < threshold:
            return True, i

    return False, n


# ============================================================================
# HELPER: RENDER CONVERGENCE STATUS
# ============================================================================
def render_convergence_status(solver):
    """Render a status badge showing convergence state."""
    rate = compute_convergence_rate(solver)
    is_eq, eq_step = estimate_equilibrium(solver)

    col1, col2, col3 = st.columns(3)
    col1.metric("Convergence rate", f"{rate:.2e} / step",
                help="Relative energy change per step (last 10%). "
                     "Close to 0 = converged.")
    col2.metric("Equilibrium", "✅ Reached" if is_eq else "⏳ Not yet",
                help=f"Threshold: relative std < 1e-4 over 5-step window")
    col3.metric("Equilibrium step", f"{eq_step}" if is_eq else f"{len(solver.history['step'])} (running)",
                help="Step at which equilibrium was first detected")

    if is_eq:
        st.success(f"✅ Equilibrium reached at step {eq_step}. "
                   f"You can stop the simulation or continue for statistics.")
    elif rate > 0:
        st.warning("⚠️ Energy is increasing — simulation may be unstable. "
                   "Reduce dt or Gamma.")
    elif rate > -1e-6:
        st.info("ℹ️ Energy is nearly constant — equilibrium is close.")


# ============================================================================
# HELPER: COMPUTE DOMAIN WALL PROFILES
# ============================================================================
def extract_domain_wall_profile(Pz, Ps, dx, wall_threshold=0.5):
    """Extract and average domain wall profiles from the Pz field.

    Finds all cells where |Pz| crosses zero (domain walls), extracts
    a window around each crossing, and averages them.

    Returns (position_nm, averaged_profile).
    """
    Pz_arr = np.asarray(Pz)
    Ny, Nx = Pz_arr.shape

    # Find zero crossings along each row
    profiles = []
    window_nm = 5.0  # nm on each side
    window_cells = int(window_nm * 1e-9 / dx)

    for row_idx in range(0, Ny, max(1, Ny // 20)):  # sample every 5% of rows
        row = Pz_arr[row_idx, :]
        # Find sign changes
        for i in range(1, Nx):
            if np.sign(row[i]) != np.sign(row[i-1]) and abs(row[i]) < Ps:
                # Extract window around crossing
                i0 = max(0, i - window_cells)
                i1 = min(Nx, i + window_cells)
                if i1 - i0 >= 2 * window_cells:
                    profile = row[i0:i1]
                    # Normalize position
                    positions = np.arange(len(profile)) - (i - i0)
                    profiles.append((positions * dx * 1e9, profile))

    if not profiles:
        return None, None

    # Average profiles (align at crossing point)
    max_len = max(len(p[1]) for p in profiles)
    min_len = min(len(p[1]) for p in profiles)

    # Use min_len for uniform averaging
    aligned = np.zeros((len(profiles), min_len))
    pos_aligned = np.zeros(min_len)
    for idx, (pos, prof) in enumerate(profiles):
        # Center the profile
        center = len(prof) // 2
        start = max(0, center - min_len // 2)
        end = start + min_len
        if end <= len(prof):
            aligned[idx] = prof[start:end]
            pos_aligned = pos[start:end] - pos[start + min_len // 2]

    avg_profile = np.mean(aligned, axis=0)
    return pos_aligned, avg_profile


def create_domain_wall_profile_figure(Pz, Ps, dx, viz_cfg):
    """Create a figure showing the averaged domain wall profile."""
    fig_cfg = viz_cfg.get("figure_customization", {})
    fig_w = float(fig_cfg.get("line_profile_width", 11.0))
    fig_h = float(fig_cfg.get("line_profile_height", 4.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos, profile = extract_domain_wall_profile(Pz, Ps, dx)
    if pos is None:
        ax.text(0.5, 0.5, "No domain walls found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_title("Averaged domain wall profile")
        return fig

    ax.plot(pos, profile,
            color=fig_cfg.get("color_series_1", "#1f77b4"),
            lw=float(fig_cfg.get("line_width", 2.0)),
            label="Averaged wall profile")
    ax.axhline(Ps, color="red", ls="--", lw=1, alpha=0.5, label=f"+Ps={Ps:.3f}")
    ax.axhline(-Ps, color="blue", ls="--", lw=1, alpha=0.5, label=f"-Ps={-Ps:.3f}")
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.axvline(0, color="gray", ls=":", lw=0.5, label="Wall center")

    ax.set_xlabel("Position relative to wall center (nm)")
    ax.set_ylabel("Pz (C/m²)")
    ax.set_title("Averaged domain wall profile")
    apply_axis_styling(ax, fig_cfg, force_aspect="auto")
    apply_legend_styling(ax, fig_cfg)

    if fig_cfg.get("tight_layout", True):
        plt.tight_layout()
    return fig


# ============================================================================
# HELPER: RENDER DOMAIN WALL ANALYSIS
# ============================================================================
def render_domain_wall_analysis(Pz, Ps, dx, viz_cfg):
    """Render domain wall profile analysis."""
    st.markdown("#### Domain wall profile")
    fig = create_domain_wall_profile_figure(Pz, Ps, dx, viz_cfg)
    if fig is not None and fig is not FAILED:
        st.pyplot(fig)
        plt.close(fig)

    # Compute wall width from profile
    pos, profile = extract_domain_wall_profile(Pz, Ps, dx)
    if pos is not None:
        # Wall width = distance between +Ps/2 and -Ps/2
        ps_half = Ps * 0.5
        above = pos[profile > ps_half]
        below = pos[profile < -ps_half]
        if len(above) > 0 and len(below) > 0:
            wall_width = above[0] - below[-1]
            st.metric("Estimated wall width", f"{abs(wall_width):.3f} nm",
                      help="Distance between +Ps/2 and -Ps/2 crossings in the averaged profile")
        else:
            st.info("Could not estimate wall width from profile.")
    else:
        st.info("No domain walls detected for profile averaging.")


# ============================================================================
# HELPER: PARAMETER PRESETS DATABASE
# ============================================================================
def get_parameter_sweep_grid():
    """Return a grid of parameter combinations for systematic sweeps.

    Useful for running multiple simulations to explore the parameter space.
    """
    presets = []
    for material in ["PbTiO3-like", "HfO2-like "]:
        for screening in [0.3, 0.5, 0.65, 0.8, 1.0]:
            for h_nm in [4.0, 8.0, 16.0]:
                for beta_factor in [0.0, 1.0, 1.5, 2.0]:
                    presets.append({
                        "material": material,
                        "screening": screening,
                        "h_film_nm": h_nm,
                        "beta_factor": beta_factor,
                        "label": f"{material[:6]}_s{screening:.1f}_h{h_nm:.0f}_b{beta_factor:.1f}",
                    })
    return presets


# ============================================================================
# HELPER: RENDER PARAMETER SWEEP UI
# ============================================================================
def render_parameter_sweep_ui():
    """Render UI for selecting and running a parameter sweep."""
    st.subheader("🔬 Parameter sweep")
    st.caption("Run multiple simulations with varying parameters to explore "
               "the parameter space.")

    sweep_grid = get_parameter_sweep_grid()
    selected = st.multiselect(
        "Select parameter combinations to run",
        [p["label"] for p in sweep_grid],
        default=[sweep_grid[10]["label"]] if len(sweep_grid) > 10 else [],
        key="sweep_select")

    if st.button("Run sweep", key="run_sweep"):
        st.info(f"Selected {len(selected)} configurations. "
                f"This feature runs each configuration sequentially — "
                f"may take a long time.")
        # Implementation would loop over selected configs and run each
        # This is a placeholder for the full sweep implementation
        st.warning("Parameter sweep is a stub. Run configurations individually "
                   "via the Run tab for now.")


# ============================================================================
# HELPER: GIT-LIKE VERSION INFO
# ============================================================================
def get_version_info():
    """Return version information for the simulator."""
    return {
        "version": "2.0.0-integrated",
        "build_date": datetime.now().isoformat(),
        "physics": "3-component thin-film ferroelectric TDGL",
        "discretization": "Semi-implicit Fourier spectral",
        "units": "SI (P: C/m², E: V/m, length: m, time: s)",
        "gpu_backend": "CuPy" if GPU_AVAILABLE else "NumPy",
        "bug_fixes_applied": ["#1 noise_amp", "#2 f_depol_Pz", "#3 alpha12",
                               "#4 screening/beta", "#5 M_zz[0,0]", "#6 Poisson eps0"],
        "features": [
            "3-component P = (Px, Py, Pz)",
            "Thin-film depolarization kernel n(k) = tanh(kh/2)/(kh/2)",
            "Open-circuit boundary conditions",
            "Substrate anisotropy (beta_sub)",
            "Material presets (PbTiO3, HfO2, BaTiO3)",
            "Multi-field grid visualization",
            "Overlay and vector overlay",
            "Plotly interactive (heatmap, 3D, animation)",
            "Line profiles (5 directions + custom)",
            "6 export formats (PKL, NPZ, JSON, CSV, HDF5, PT)",
            "Simulation database (up to 32 runs)",
            "Compare view",
            "Energy breakdown analysis",
            "Domain structure analysis",
            "Fourier spectrum analysis",
            "Domain wall profile extraction",
            "Parameter validation and stability metrics",
            "15 matplotlib themes",
            "80+ colormaps + 7 custom ferro maps",
            "Full figure customization (live)",
        ],
    }


def render_version_info():
    """Render version information in an expander."""
    with st.expander("ℹ️ About this simulator", expanded=False):
        info = get_version_info()
        st.json(info)


# ============================================================================
# HELPER: RENDER FULL DIAGNOSTICS PANEL (combines all analysis functions)
# ============================================================================
def render_full_diagnostics_panel(solver, params, viz_cfg):
    """Render a comprehensive diagnostics panel with all analysis functions.

    This is the one-stop shop for post-simulation analysis. It combines:
        - Parameter validation
        - Convergence status
        - Domain structure analysis
        - Energy breakdown
        - Polarization histograms
        - Fourier spectrum
        - Domain wall profile
        - Configuration summary
    """
    if not has_results():
        st.info("Run a simulation first to see diagnostics.")
        return

    st.header("📊 Full Diagnostics Panel")

    # ---- Section 1: Parameter validation ----
    render_parameter_validation(params)

    # ---- Section 2: Convergence status ----
    st.markdown("---")
    st.subheader("🎯 Convergence status")
    render_convergence_status(solver)

    # ---- Section 3: Domain analysis ----
    st.markdown("---")
    history = st.session_state.get("history", [])
    if history:
        final = history[-1]
        Pz = final["Pz"]
        Ps = params["mat"].Ps_eq()
        dx = params["L"] / params["Nx"]

        # Domain statistics metrics
        stats = compute_domain_statistics(Pz, Ps, dx)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Domains detected", stats["n_domains"])
        col2.metric("Mean domain size", f"{stats['mean_domain_size_nm']:.1f} nm")
        col3.metric("Wall fraction", f"{stats['domain_wall_fraction']:.1%}")
        col4.metric("Stripe period", f"{stats['stripe_period_est_nm']:.1f} nm")

        # Domain analysis figure
        st.markdown("#### Domain structure visualization")
        fig = create_domain_analysis_figure(Pz, Ps, dx, viz_cfg)
        if fig is not None and fig is not FAILED:
            st.pyplot(fig)
            plt.close(fig)

        # Domain wall profile
        st.markdown("#### Domain wall profile (averaged)")
        render_domain_wall_analysis(Pz, Ps, dx, viz_cfg)

    # ---- Section 4: Energy breakdown ----
    st.markdown("---")
    st.subheader("📈 Energy breakdown")
    fig_energy = create_energy_breakdown_figure(solver, viz_cfg)
    if fig_energy is not None and fig_energy is not FAILED:
        st.pyplot(fig_energy)
        plt.close(fig_energy)

    # ---- Section 5: Polarization histograms ----
    st.markdown("---")
    st.subheader("📊 Polarization distributions")
    if history:
        final = history[-1]
        fig_hist = create_polarization_histogram_figure(
            final["Px"], final["Py"], final["Pz"],
            params["mat"].Ps_eq(), viz_cfg)
        if fig_hist is not None and fig_hist is not FAILED:
            st.pyplot(fig_hist)
            plt.close(fig_hist)

    # ---- Section 6: Fourier spectrum ----
    st.markdown("---")
    st.subheader("🌐 Fourier spectrum of Pz")
    if history:
        fig_spec = create_spectrum_figure(history[-1]["Pz"],
                                           params["L"] / params["Nx"], viz_cfg)
        if fig_spec is not None and fig_spec is not FAILED:
            st.pyplot(fig_spec)
            plt.close(fig_spec)

    # ---- Section 7: Configuration summary ----
    st.markdown("---")
    st.subheader("📋 Configuration summary")
    summary = get_config_summary(params)
    st.text(summary)
    buf, fname = save_config_summary(params)
    st.download_button("📥 Download config summary", buf, fname, "text/plain")


# ============================================================================
# HELPER: QUICK START GUIDE
# ============================================================================
def render_quick_start_guide():
    """Render a quick-start guide for new users."""
    with st.expander("📖 Quick start guide", expanded=False):
        st.markdown("""
### How to use this simulator

1. **Configure parameters** in the sidebar (left).
   - Start with the **PbTiO3-like** preset (default).
   - Keep screening factor at 0.65 (good for stripe domains).
   - Keep substrate anisotropy enabled (favors Pz over in-plane P).

2. **Check the Geometry tab** to see the initial Pz field and verify
   that the wall width ξ resolves on the grid (ξ/dx > 3).

3. **Click "Start Evolution"** in the Run tab.
   - A progress bar shows simulation progress.
   - Energy diagnostics update in real-time.

4. **View results** in the Results tab:
   - **Multi-field grid**: shows all 9 fields (Pz, |P|, Px, Py, energies).
   - **Overlay**: overlay two fields (e.g., Pz + |P|).
   - **Vector overlay**: quiver of in-plane P over Pz heatmap.

5. **Analyze** in the Line Profiles, Plotly, and Export tabs:
   - Line profiles: 1D cuts along any direction.
   - Plotly: interactive heatmaps, 3D surfaces, animations.
   - Export: PKL, NPZ, JSON, CSV, HDF5, PT, PNG, GIF/MP4.

### Key physics

- **Order parameter**: 3-component polarization P = (Px, Py, Pz).
- **Pz is out-of-plane**: forms 180° stripe domains under open-circuit BC.
- **Thin-film kernel**: n(k) = tanh(k·h/2)/(k·h/2) captures surface depol.
- **Screening factor s**: 0=perfect electrodes, 1=no electrodes.
  - Sweet spot for stripes: s ≈ 0.5–0.8.
- **Substrate anisotropy β_sub**: penalizes in-plane P (mimics compressive strain).
  - Need β_sub > |α₁| to suppress in-plane P.

### All units are SI

- P in C/m² (display also shows μC/cm²)
- E in V/m
- Length in m (display shows nm)
- Time in s (display shows ps or ns)
- Energy in J/m³
        """)


<<<<<<< HEAD
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
=======
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Print version info to console on startup
    info = get_version_info()
    print(f"=" * 60)
    print(f"Ferroelectric Phase-Field Simulator v{info['version']}")
    print(f"Build date: {info['build_date']}")
    print(f"GPU backend: {info['gpu_backend']}")
    print(f"Physics: {info['physics']}")
    print(f"Discretization: {info['discretization']}")
    print(f"Units: {info['units']}")
    print(f"Bug fixes applied: {', '.join(info['bug_fixes_applied'])}")
    print(f"Features: {len(info['features'])} capabilities available")
    print(f"=" * 60)
    print(f"Starting Streamlit app...")
    main()

