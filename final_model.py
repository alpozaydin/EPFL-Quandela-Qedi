#!/usr/bin/env python3
"""
Final Model: Quantum Reservoir Computing for Swaption Pricing
=============================================================
Train on historical data, predict future swaption prices.

Pipeline:
  Raw prices → StandardScaler → PCA(5) → sliding window(5) →
  5× Perceval QRC (8m/3ph, UNBUNCHED, LexGrouping(10), fixed) + raw →
  Ridge → PCA inverse → Scaler inverse → prices
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import perceval as pcvl
from merlin import QuantumLayer, ComputationSpace, LexGrouping

# ── Config ───────────────────────────────────────────────────────────────
N_PCA       = 5
WINDOW      = 5
N_MODES     = 8
N_PHOTONS   = 3
N_ENCODE    = 5
N_LAYERS    = 5
N_RESERVOIRS = 5
RIDGE_ALPHA = 0.1
LEX_OUT     = 10     # LexGrouping output dimension per reservoir

TOTAL_ENC   = N_ENCODE * N_LAYERS  # 25
INPUT_STATE = [1] * N_PHOTONS + [0] * (N_MODES - N_PHOTONS)

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "CHALLENGE RESOURCES" / "DATASETS"


# =====================================================================
# BUILDING BLOCKS
# =====================================================================

def build_circuit(n_modes, n_encode, n_layers, seed):
    """Build one hand-crafted Perceval photonic circuit."""
    circuit = pcvl.Circuit(n_modes)
    c = 0
    for li in range(n_layers):
        circuit.add(0, pcvl.GenericInterferometer(
            n_modes,
            lambda i, _l=li: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_l}_i{i}"))
                              // pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_l}_o{i}"))),
            shape=pcvl.InterferometerShape.RECTANGLE,
        ))
        for m in range(n_encode):
            circuit.add(m, pcvl.PS(pcvl.P(f"input{c}")))
            c += 1
    circuit.add(0, pcvl.GenericInterferometer(
        n_modes,
        lambda i: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_f_i{i}"))
                   // pcvl.BS() // pcvl.PS(pcvl.P(f"t_f_o{i}"))),
        shape=pcvl.InterferometerShape.RECTANGLE,
    ))
    return circuit, c


def build_reservoirs():
    """Build ensemble of fixed-random quantum reservoirs with LexGrouping."""
    reservoirs = []
    for r in range(N_RESERVOIRS):
        torch.manual_seed(42 + r * 1000)
        circ, n_enc = build_circuit(N_MODES, N_ENCODE, N_LAYERS, seed=42 + r * 1000)
        core = QuantumLayer(
            input_size=n_enc, circuit=circ, input_state=INPUT_STATE,
            input_parameters=["input"], trainable_parameters=["t"],
            computation_space=ComputationSpace.UNBUNCHED,
            dtype=torch.float32,
        )
        layer = nn.Sequential(core, LexGrouping(core.output_size, LEX_OUT))
        layer.eval()
        reservoirs.append(layer)
    return reservoirs


def quantum_features(x, reservoirs):
    """Pass a single 25-dim vector through all reservoirs → concatenated output."""
    if len(x) < TOTAL_ENC:
        x = np.pad(x, (0, TOTAL_ENC - len(x)))
    elif len(x) > TOTAL_ENC:
        x = x[:TOTAL_ENC]
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return np.concatenate([r(xt).squeeze(0).numpy() for r in reservoirs])


def make_windows(pc_data):
    """Create (X, y) pairs from PCA time series."""
    X, y = [], []
    for t in range(WINDOW, len(pc_data)):
        X.append(pc_data[t - WINDOW: t].flatten())
        y.append(pc_data[t])
    return np.array(X), np.array(y)


# =====================================================================
# TRAIN
# =====================================================================

def train(prices_train):
    """
    Train the full pipeline on raw price data.

    Args:
        prices_train: np.array of shape (N_days, 224)

    Returns:
        model dict with all fitted components + training diagnostics
    """
    # 1. Fit scaler & PCA
    scaler = StandardScaler().fit(prices_train)
    scaled = scaler.transform(prices_train)
    pca = PCA(n_components=N_PCA).fit(scaled)
    pc = pca.transform(scaled)

    # 2. Build reservoirs
    reservoirs = build_reservoirs()

    # 3. Create windows
    X, y = make_windows(pc)

    # 4. Extract features (quantum + raw)
    Q = np.array([quantum_features(x, reservoirs) for x in X])
    features = np.hstack([Q, X])

    # 5. Fit Ridge
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(features, y)

    # 6. Training predictions (for diagnostics)
    y_train_pred_pca = ridge.predict(features)
    y_train_pred_prices = scaler.inverse_transform(pca.inverse_transform(y_train_pred_pca))
    y_train_true_prices = scaler.inverse_transform(pca.inverse_transform(y))

    return {
        "scaler": scaler,
        "pca": pca,
        "reservoirs": reservoirs,
        "ridge": ridge,
        "last_pc": pc[-WINDOW:],  # last window for inference
        # diagnostics
        "train_true_pca": y,
        "train_pred_pca": y_train_pred_pca,
        "train_true_prices": y_train_true_prices,
        "train_pred_prices": y_train_pred_prices,
        "pca_explained_var": pca.explained_variance_ratio_,
        "ridge_coef": ridge.coef_,
    }


# =====================================================================
# PREDICT
# =====================================================================

def predict(model, n_days):
    """
    Predict n_days into the future autoregressively.

    Args:
        model: dict from train()
        n_days: how many future days to predict

    Returns:
        np.array of shape (n_days, 224) — predicted prices
    """
    buffer = list(model["last_pc"])
    reservoirs = model["reservoirs"]
    ridge = model["ridge"]
    preds_pca = []

    for _ in range(n_days):
        window = np.array(buffer[-WINDOW:]).flatten()
        q = quantum_features(window, reservoirs)
        feat = np.concatenate([q, window]).reshape(1, -1)
        pred = ridge.predict(feat)[0]
        preds_pca.append(pred)
        buffer.append(pred)

    preds_pca = np.array(preds_pca)
    scaled = model["pca"].inverse_transform(preds_pca)
    prices = model["scaler"].inverse_transform(scaled)
    return prices


# =====================================================================
# MAIN
# =====================================================================

def generate_plots(model, actual, predicted, features, train_days, pred_days, out):
    """Generate all diagnostic and inference plots."""

    # ── Plot 1: PCA Variance Explained ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    var = model["pca_explained_var"]
    cumvar = np.cumsum(var)
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var * 100, color="#3498db", alpha=0.7, label="Individual")
    ax.plot(x, cumvar * 100, "o-", color="#e74c3c", linewidth=2, label="Cumulative")
    for i, cv in enumerate(cumvar):
        ax.annotate(f"{cv*100:.1f}%", (x[i], cv*100 + 1), ha="center", fontsize=9)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Variance Explained")
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out / "pca_variance.png", dpi=150)
    plt.close(fig)
    print("  [1/8] PCA variance plot saved")

    # ── Plot 2: Training Fit — PCA space (per component) ─────────────────
    y_true_pca = model["train_true_pca"]
    y_pred_pca = model["train_pred_pca"]
    n_comp = y_true_pca.shape[1]

    fig, axes = plt.subplots(n_comp, 1, figsize=(14, 3 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]
    for c in range(n_comp):
        ax = axes[c]
        ax.plot(y_true_pca[:, c], color="#2c3e50", linewidth=0.8, label="Actual", alpha=0.8)
        ax.plot(y_pred_pca[:, c], color="#e74c3c", linewidth=0.8, label="Predicted", alpha=0.7)
        r2 = r2_score(y_true_pca[:, c], y_pred_pca[:, c])
        ax.set_ylabel(f"PC{c+1}")
        ax.set_title(f"PC{c+1}  (train R² = {r2:.6f})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Training sample index")
    fig.suptitle("Training Fit in PCA Space", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "training_fit_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] Training fit (PCA) plot saved")

    # ── Plot 3: Training Fit — Price space (sample features) ─────────────
    sample_idx = np.linspace(0, len(features) - 1, 6, dtype=int)
    y_true_px = model["train_true_prices"]
    y_pred_px = model["train_pred_prices"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for k, (ax, si) in enumerate(zip(axes.flat, sample_idx)):
        ax.plot(y_true_px[:, si], color="#2c3e50", linewidth=0.8, label="Actual")
        ax.plot(y_pred_px[:, si], color="#e74c3c", linewidth=0.8, alpha=0.7, label="Predicted")
        r2 = r2_score(y_true_px[:, si], y_pred_px[:, si])
        ax.set_title(f"{features[si]}  (R²={r2:.4f})", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Training Fit — Sample Swaption Prices", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "training_fit_prices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/8] Training fit (prices) plot saved")

    # ── Plot 4: Training Residual Distribution ───────────────────────────
    residuals = (y_true_px - y_pred_px).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(residuals, bins=80, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Training Residuals  (mean={np.mean(residuals):.2e}, std={np.std(residuals):.2e})")
    axes[1].hist(np.abs(residuals), bins=80, color="#e67e22", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("|Residual|")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Absolute Residuals  (median={np.median(np.abs(residuals)):.2e})")
    fig.tight_layout()
    fig.savefig(out / "training_residuals.png", dpi=150)
    plt.close(fig)
    print("  [4/8] Training residuals plot saved")

    # ── Plot 5: Inference — Predicted vs Actual Scatter ──────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual.flatten(), predicted.flatten(), s=3, alpha=0.4, color="#3498db")
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    r2 = r2_score(actual.flatten(), predicted.flatten())
    mse = mean_squared_error(actual.flatten(), predicted.flatten())
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Inference: Predicted vs Actual (Days {train_days+1}-{train_days+pred_days})\n"
                 f"R² = {r2:.6f}   MSE = {mse:.2e}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "inference_scatter.png", dpi=150)
    plt.close(fig)
    print("  [5/8] Inference scatter plot saved")

    # ── Plot 6: Inference — Error Heatmap ────────────────────────────────
    errors = predicted - actual  # (PRED_DAYS, 224)
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(errors, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(errors)), vmax=np.max(np.abs(errors)))
    ax.set_yticks(range(pred_days))
    ax.set_yticklabels([f"Day {train_days+d+1}" for d in range(pred_days)])
    ax.set_xlabel("Feature index (224 swaption columns)")
    ax.set_title("Prediction Error Heatmap (Predicted − Actual)")
    plt.colorbar(im, ax=ax, label="Error", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "inference_error_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [6/8] Inference error heatmap saved")

    # ── Plot 7: Inference — Per-day MSE bar chart ────────────────────────
    day_mse = [mean_squared_error(actual[d], predicted[d]) for d in range(pred_days)]
    day_mae = [mean_absolute_error(actual[d], predicted[d]) for d in range(pred_days)]
    day_labels = [f"Day {train_days+d+1}" for d in range(pred_days)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, pred_days))
    axes[0].bar(day_labels, day_mse, color=colors, edgecolor="white")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Per-Day MSE")
    for i, v in enumerate(day_mse):
        axes[0].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(day_labels, day_mae, color=colors, edgecolor="white")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Per-Day MAE")
    for i, v in enumerate(day_mae):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Autoregressive Inference Error (Days {train_days+1}-{train_days+pred_days})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "inference_per_day_error.png", dpi=150)
    plt.close(fig)
    print("  [7/8] Per-day error bar chart saved")

    # ── Plot 8: Tenor × Maturity MAE Surface Heatmaps ────────────────────
    # Parse tenor/maturity from feature column names
    def _parse_col(col):
        parts = col.split(";")
        tenor    = float(parts[0].split(":")[1].strip())
        maturity = float(parts[1].split(":")[1].strip())
        return tenor, maturity

    tm_map = {c: _parse_col(c) for c in features}
    tenors     = sorted({t for t, _ in tm_map.values()})
    maturities = sorted({m for _, m in tm_map.values()})
    n_ten, n_mat = len(tenors), len(maturities)

    # Helper: build a surface from per-feature absolute errors for given days
    def _build_surface(abs_errs):
        """abs_errs: shape (n_features,) — already averaged over desired days."""
        surface = np.full((n_ten, n_mat), np.nan)
        for fi, feat in enumerate(features):
            t, m = tm_map[feat]
            surface[tenors.index(t), maturities.index(m)] = abs_errs[fi]
        return surface

    # Build per-day surfaces + average
    per_day_surfaces = []
    for d in range(pred_days):
        per_day_surfaces.append(_build_surface(np.abs(actual[d] - predicted[d])))
    avg_surface = _build_surface(np.mean(np.abs(actual - predicted), axis=0))

    # Global color range: 0 → max across ALL surfaces (for consistent comparison)
    global_max = max(np.nanmax(avg_surface),
                     max(np.nanmax(s) for s in per_day_surfaces))

    # Tick labels
    x_labels = [f"{m:.2f}" if m < 1 else f"{m:.1f}" for m in maturities]
    y_labels = [f"{int(t)}Y" if t == int(t) else f"{t}Y" for t in tenors]

    def _draw_heatmap(ax, surface, title, vmax, annotate=True):
        im = ax.imshow(surface, cmap="plasma", aspect="auto",
                       interpolation="bilinear", origin="lower",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(n_mat))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_ten))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_ylabel("Tenor (years)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if annotate:
            for ti in range(n_ten):
                for mi in range(n_mat):
                    val = surface[ti, mi]
                    if not np.isnan(val):
                        text_color = "white" if val > 0.55 * vmax else "black"
                        ax.text(mi, ti, f"{val:.4f}", ha="center", va="center",
                                fontsize=4.5, color=text_color)
        return im

    # ── 8a: Average + 3D ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 8),
                              gridspec_kw={"width_ratios": [1, 1]})

    im = _draw_heatmap(axes[0], avg_surface,
                       f"MAE — Averaged over Days {train_days+1}–{train_days+pred_days}",
                       global_max, annotate=True)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.02)
    cbar.set_label("MAE (price units)", fontsize=10)
    cbar.set_ticks([0, global_max / 4, global_max / 2, 3 * global_max / 4, global_max])
    cbar.set_ticklabels([f"{v:.5f}" for v in
                         [0, global_max/4, global_max/2, 3*global_max/4, global_max]])
    cbar.ax.set_ylabel("0 = perfect  ←  MAE  →  worst", fontsize=8,
                       rotation=270, labelpad=18)

    # 3D surface
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    M, T = np.meshgrid(range(n_mat), range(n_ten))
    surf = ax3d.plot_surface(M, T, avg_surface, cmap="plasma",
                              edgecolor="none", alpha=0.9, antialiased=True,
                              vmin=0, vmax=global_max)
    ax3d.set_xlabel("Maturity", fontsize=9, labelpad=8)
    ax3d.set_ylabel("Tenor", fontsize=9, labelpad=8)
    ax3d.set_zlabel("MAE", fontsize=9, labelpad=8)
    ax3d.set_zlim(0, global_max * 1.05)
    ax3d.set_title("3D Error Surface (Average)", fontsize=12,
                   fontweight="bold", pad=15)
    ax3d.set_xticks(range(0, n_mat, 3))
    ax3d.set_xticklabels([f"{maturities[i]:.1f}" for i in range(0, n_mat, 3)],
                         fontsize=7)
    ax3d.set_yticks(range(0, n_ten, 2))
    ax3d.set_yticklabels([f"{int(tenors[i])}Y" for i in range(0, n_ten, 2)],
                         fontsize=7)
    ax3d.view_init(elev=25, azim=-60)
    cb3 = fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.1)
    cb3.set_ticks([0, global_max])
    cb3.set_ticklabels(["0 (perfect)", f"{global_max:.5f} (max)"])

    fig.suptitle("Inference Error Surface — Tenor vs Maturity (Averaged)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "tenor_maturity_mae_surface.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("  [8a/8] Tenor × Maturity MAE surface (average) saved")

    # ── 8b: All individual days ──────────────────────────────────────────
    n_cols = 3
    n_rows = int(np.ceil(pred_days / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = np.atleast_2d(axes)

    for d in range(pred_days):
        row, col = divmod(d, n_cols)
        ax = axes[row, col]
        day_label = f"Day {train_days + d + 1}"
        day_mse = mean_squared_error(actual[d], predicted[d])
        day_mae_val = mean_absolute_error(actual[d], predicted[d])
        im = _draw_heatmap(ax, per_day_surfaces[d],
                           f"{day_label}  (MAE={day_mae_val:.5f}, MSE={day_mse:.2e})",
                           global_max, annotate=True)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_ticks([0, global_max])
        cbar.set_ticklabels(["0", f"{global_max:.4f}"])

    # Hide unused subplots
    for d in range(pred_days, n_rows * n_cols):
        row, col = divmod(d, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(f"Inference Error Surface — Each Day Separately\n"
                 f"(color scale: 0 = deep purple → {global_max:.5f} = bright yellow)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "tenor_maturity_mae_per_day.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("  [8b/8] Tenor × Maturity MAE surface (per-day) saved")


def visualize_circuit(out):
    """Generate publication-quality circuit and pipeline diagrams."""

    # ── 1. Perceval circuit rendering (simplified 2-layer for clarity) ──
    print("  [C1] Rendering Perceval circuit (simplified 2-layer)...")
    saved_figs = []
    original_show = plt.show
    def capture_show(*args, **kwargs):
        for fignum in plt.get_fignums():
            saved_figs.append(plt.figure(fignum))
    plt.show = capture_show

    # Build a 2-layer version for readability
    circ_small, _ = build_circuit(N_MODES, N_ENCODE, 2, seed=42)
    pcvl.pdisplay(circ_small, output_format=pcvl.Format.MPLOT)
    if saved_figs:
        saved_figs[-1].savefig(out / "circuit_perceval_2layer.png",
                               dpi=200, bbox_inches="tight",
                               facecolor="white", edgecolor="none")
    plt.close("all")
    saved_figs.clear()

    # Full 5-layer circuit
    print("  [C2] Rendering Perceval circuit (full 5-layer)...")
    circ_full, _ = build_circuit(N_MODES, N_ENCODE, N_LAYERS, seed=42)
    pcvl.pdisplay(circ_full, output_format=pcvl.Format.MPLOT)
    if saved_figs:
        saved_figs[-1].savefig(out / "circuit_perceval_full.png",
                               dpi=200, bbox_inches="tight",
                               facecolor="white", edgecolor="none")
    plt.close("all")
    plt.show = original_show

    # ── 2. Schematic architecture diagram ─────────────────────────────
    print("  [C3] Creating architecture schematic...")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(8, 9.5, "Quantum Reservoir Computing — Photonic Circuit Architecture",
            ha="center", fontsize=16, fontweight="bold", color="#2c3e50")

    # ── Draw 8 horizontal mode lines ──────────────────────────────────
    mode_y = np.linspace(7.5, 1.5, N_MODES)
    mode_labels = [f"Mode {i}" for i in range(N_MODES)]
    photon_modes = list(range(N_PHOTONS))
    encode_modes = list(range(N_ENCODE))

    for i, y in enumerate(mode_y):
        color = "#e74c3c" if i in photon_modes else "#bdc3c7"
        lw = 2.5 if i in photon_modes else 1.5
        ax.plot([1.5, 15.5], [y, y], color=color, linewidth=lw, zorder=1)
        # Input state label
        state = "| 1 ⟩" if i in photon_modes else "| 0 ⟩"
        ax.text(1.2, y, state, ha="right", va="center", fontsize=10,
                fontfamily="serif", color="#e74c3c" if i in photon_modes else "#7f8c8d")
        ax.text(0.3, y, f"m{i}", ha="center", va="center", fontsize=9,
                fontweight="bold", color="#2c3e50")

    # ── Draw blocks for each layer ────────────────────────────────────
    block_xs = [2.5, 5.0, 7.5, 10.0, 12.5]  # 5 layers
    block_w = 1.8

    for li, bx in enumerate(block_xs):
        # Interferometer block (full modes)
        rect = plt.Rectangle((bx - block_w/2, mode_y[-1] - 0.4),
                              block_w, mode_y[0] - mode_y[-1] + 0.8,
                              facecolor="#3498db", alpha=0.15, edgecolor="#3498db",
                              linewidth=1.5, zorder=2, linestyle="-")
        ax.add_patch(rect)
        ax.text(bx, mode_y[-1] - 0.7, f"U{li+1}", ha="center", va="top",
                fontsize=9, color="#3498db", fontweight="bold")

        # Input encoding phase shifters (only on modes 0-4)
        enc_x = bx + block_w/2 + 0.3
        for m in range(N_ENCODE):
            y = mode_y[m]
            # Phase shifter symbol (small box)
            ps_rect = plt.Rectangle((enc_x - 0.15, y - 0.2), 0.3, 0.4,
                                     facecolor="#e67e22", alpha=0.8,
                                     edgecolor="#d35400", linewidth=1.2, zorder=3)
            ax.add_patch(ps_rect)
            ax.text(enc_x, y, "φ", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold", zorder=4)

        # Label the encoding
        if li == 0:
            ax.annotate("Input\nencoding",
                        xy=(enc_x, mode_y[N_ENCODE-1] - 0.5),
                        xytext=(enc_x, mode_y[N_ENCODE-1] - 1.2),
                        ha="center", fontsize=8, color="#d35400",
                        arrowprops=dict(arrowstyle="->", color="#d35400"))

    # Final interferometer
    fx = 14.5
    rect = plt.Rectangle((fx - block_w/2, mode_y[-1] - 0.4),
                          block_w, mode_y[0] - mode_y[-1] + 0.8,
                          facecolor="#9b59b6", alpha=0.15, edgecolor="#9b59b6",
                          linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(fx, mode_y[-1] - 0.7, "U_final", ha="center", va="top",
            fontsize=9, color="#9b59b6", fontweight="bold")

    # Measurement symbol
    ax.text(16.0, mode_y[3], "📊", ha="center", va="center", fontsize=20, zorder=5)
    ax.text(16.0, mode_y[3] - 0.8, "Fock probs\n(UNBUNCHED)", ha="center",
            va="top", fontsize=8, color="#7f8c8d")

    # Legend
    legend_y = 0.3
    ax.plot([1.5, 2.2], [legend_y, legend_y], color="#e74c3c", linewidth=2.5)
    ax.text(2.4, legend_y, "Photon-occupied mode", va="center", fontsize=9, color="#e74c3c")
    ax.plot([6.0, 6.7], [legend_y, legend_y], color="#bdc3c7", linewidth=1.5)
    ax.text(6.9, legend_y, "Vacuum mode", va="center", fontsize=9, color="#7f8c8d")
    ps_r = plt.Rectangle((10.5, legend_y - 0.15), 0.3, 0.3,
                           facecolor="#e67e22", edgecolor="#d35400", linewidth=1)
    ax.add_patch(ps_r)
    ax.text(11.0, legend_y, "Phase shifter (input encoding)", va="center",
            fontsize=9, color="#d35400")

    # Annotations
    ax.annotate("", xy=(2.5, 8.5), xytext=(14.5, 8.5),
                arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.5))
    ax.text(8.5, 8.7, f"{N_LAYERS} encoding layers × {N_ENCODE} PS each = "
            f"{TOTAL_ENC} input parameters",
            ha="center", fontsize=10, color="#2c3e50")

    # Bracket for memory modes
    ax.annotate("", xy=(0.7, mode_y[N_ENCODE]), xytext=(0.7, mode_y[-1]),
                arrowprops=dict(arrowstyle="-[", color="#27ae60", lw=1.5,
                               mutation_scale=10))
    ax.text(0.5, (mode_y[N_ENCODE] + mode_y[-1])/2, "No\nencoding",
            ha="right", va="center", fontsize=8, color="#27ae60", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(out / "circuit_architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # ── 3. Full pipeline diagram ──────────────────────────────────────
    print("  [C4] Creating pipeline overview...")
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_xlim(-0.5, 19)
    ax.set_ylim(-1, 4)
    ax.axis("off")

    # Pipeline boxes
    boxes = [
        (0.5, "Raw Prices\n(N × 224)", "#ecf0f1", "#7f8c8d"),
        (2.8, "Standard\nScaler", "#3498db", "white"),
        (5.1, f"PCA\n(224 → {N_PCA})", "#2ecc71", "white"),
        (7.4, f"Sliding\nWindow ({WINDOW})", "#e67e22", "white"),
        (9.7, f"5× Perceval\nQRC (8m/3ph)", "#9b59b6", "white"),
        (12.0, f"LexGrouping\n(→ 10/res)", "#e74c3c", "white"),
        (14.0, "Concat\nQ + raw", "#1abc9c", "white"),
        (16.0, f"Ridge\n(α={RIDGE_ALPHA})", "#f39c12", "white"),
        (18.0, f"PCA⁻¹ +\nScaler⁻¹", "#3498db", "white"),
    ]

    for x, label, bg, fg in boxes:
        w = 1.8
        rect = plt.Rectangle((x - w/2, 0.5), w, 2.0,
                               facecolor=bg, edgecolor="#2c3e50",
                               linewidth=1.5, zorder=2,
                               alpha=0.85 if bg != "#ecf0f1" else 0.5)
        ax.add_patch(rect)
        ax.text(x, 1.5, label, ha="center", va="center", fontsize=9,
                color=fg, fontweight="bold", zorder=3)

    # Arrows between boxes
    arrow_xs = [(boxes[i][0] + 0.9, boxes[i+1][0] - 0.9) for i in range(len(boxes) - 1)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    # Dimension annotations
    dims = [
        (0.5, "N × 224"),
        (2.8, "N × 224"),
        (5.1, f"N × {N_PCA}"),
        (7.4, f"(N-{WINDOW}) × {WINDOW*N_PCA}"),
        (9.7, f"× {N_RESERVOIRS} reservoirs"),
        (12.0, f"→ {LEX_OUT * N_RESERVOIRS} dim"),
        (14.0, f"{LEX_OUT * N_RESERVOIRS + WINDOW*N_PCA} dim"),
        (16.0, f"→ {N_PCA} PCA"),
        (18.0, "→ 224 prices"),
    ]
    for x, label in dims:
        ax.text(x, 0.2, label, ha="center", va="top", fontsize=7,
                color="#7f8c8d", fontstyle="italic")

    # Title
    ax.text(9.25, 3.5, "Full Pipeline: Quantum Reservoir Computing for Swaption Pricing",
            ha="center", fontsize=14, fontweight="bold", color="#2c3e50")

    fig.tight_layout()
    fig.savefig(out / "pipeline_overview.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # ── 4. Reservoir ensemble diagram ─────────────────────────────────
    print("  [C5] Creating reservoir ensemble diagram...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 8)
    ax.axis("off")

    ax.text(6, 7.5, f"Ensemble of {N_RESERVOIRS} Fixed-Random Quantum Reservoirs",
            ha="center", fontsize=14, fontweight="bold", color="#2c3e50")

    # Input box
    in_rect = plt.Rectangle((-0.5, 2.5), 2, 2.5,
                              facecolor="#3498db", alpha=0.2,
                              edgecolor="#3498db", linewidth=2)
    ax.add_patch(in_rect)
    ax.text(0.5, 4.3, "Input", ha="center", fontsize=11, fontweight="bold", color="#3498db")
    ax.text(0.5, 3.6, f"PCA window\n{WINDOW}×{N_PCA} = {WINDOW*N_PCA} dim",
            ha="center", fontsize=9, color="#2c3e50")

    # Reservoir boxes
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#9b59b6"]
    for r in range(N_RESERVOIRS):
        y = 6.0 - r * 1.2
        # QRC box
        rect = plt.Rectangle((3.0, y - 0.4), 3.5, 0.8,
                               facecolor=colors[r], alpha=0.2,
                               edgecolor=colors[r], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(4.75, y, f"QRC #{r+1}  (seed={42 + r*1000})",
                ha="center", va="center", fontsize=9, color=colors[r],
                fontweight="bold")
        ax.text(4.75, y - 0.25, f"{N_MODES}m / {N_PHOTONS}ph / UNBUNCHED",
                ha="center", va="center", fontsize=7, color="#7f8c8d")

        # Arrow from input
        ax.annotate("", xy=(3.0, y), xytext=(1.5, 3.75),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1,
                                   connectionstyle="arc3,rad=0.1"))

        # LexGrouping box
        lex_rect = plt.Rectangle((7.0, y - 0.35), 1.8, 0.7,
                                   facecolor="#1abc9c", alpha=0.2,
                                   edgecolor="#1abc9c", linewidth=1)
        ax.add_patch(lex_rect)
        ax.text(7.9, y, f"Lex({LEX_OUT})", ha="center", va="center",
                fontsize=8, color="#1abc9c", fontweight="bold")

        ax.annotate("", xy=(7.0, y), xytext=(6.5, y),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1))

        # Arrow to concat
        ax.annotate("", xy=(9.8, 3.75), xytext=(8.8, y),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1,
                                   connectionstyle="arc3,rad=-0.1"))

    # Concat + Raw
    cat_rect = plt.Rectangle((9.5, 2.5), 2.5, 2.5,
                               facecolor="#f39c12", alpha=0.2,
                               edgecolor="#f39c12", linewidth=2)
    ax.add_patch(cat_rect)
    ax.text(10.75, 4.5, "Concatenate", ha="center", fontsize=11,
            fontweight="bold", color="#f39c12")
    ax.text(10.75, 3.8, f"Quantum: {LEX_OUT}×{N_RESERVOIRS}={LEX_OUT*N_RESERVOIRS}",
            ha="center", fontsize=9, color="#2c3e50")
    ax.text(10.75, 3.3, f"Raw PCA: {WINDOW*N_PCA}",
            ha="center", fontsize=9, color="#2c3e50")
    ax.text(10.75, 2.8, f"Total: {LEX_OUT*N_RESERVOIRS + WINDOW*N_PCA} dim",
            ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

    # Raw skip connection
    ax.annotate("", xy=(9.5, 3.0), xytext=(1.5, 3.0),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=2,
                                linestyle="--"))
    ax.text(5.5, 2.6, "Raw PCA features (skip connection)",
            ha="center", fontsize=8, color="#3498db", fontstyle="italic")

    # Ridge output
    ax.annotate("→ Ridge → 5 PCA → 224 prices",
                xy=(12.0, 3.75), fontsize=11, color="#2c3e50",
                fontweight="bold", va="center")

    fig.tight_layout()
    fig.savefig(out / "reservoir_ensemble.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    print("  Circuit visualizations complete!")


if __name__ == "__main__":
    # Load data
    df = pd.read_excel(DATA / "train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    features = [c for c in df.columns if c != "Date"]
    prices = df[features].values.astype(np.float64)

    # --- Config ---
    TRAIN_DAYS = 450
    PRED_DAYS  = 6

    print(f"Training on days 1-{TRAIN_DAYS}...")
    model = train(prices[:TRAIN_DAYS])
    print("Done.\n")

    print(f"Predicting days {TRAIN_DAYS+1}-{TRAIN_DAYS+PRED_DAYS}...")
    predicted = predict(model, PRED_DAYS)
    actual = prices[TRAIN_DAYS:TRAIN_DAYS+PRED_DAYS]

    # Metrics table
    print(f"\n{'Day':<6} {'MSE':>14} {'MAE':>10} {'RelErr%':>10}")
    print("─" * 42)
    for d in range(PRED_DAYS):
        mse = mean_squared_error(actual[d], predicted[d])
        mae = mean_absolute_error(actual[d], predicted[d])
        rel = 100 * np.mean(np.abs(actual[d] - predicted[d]) / (np.abs(actual[d]) + 1e-10))
        print(f"{TRAIN_DAYS+d+1:<6} {mse:>14.2e} {mae:>10.6f} {rel:>9.4f}%")

    overall_mse = mean_squared_error(actual.flatten(), predicted.flatten())
    overall_r2 = r2_score(actual.flatten(), predicted.flatten())
    print(f"\nOverall MSE: {overall_mse:.2e}")
    print(f"Overall R² : {overall_r2:.6f}")

    # Save CSVs
    out = BASE / "quantum_results" / "final"
    out.mkdir(parents=True, exist_ok=True)

    for label, data in [("actual", actual), ("predicted", predicted)]:
        pd.DataFrame(data, columns=features,
                     index=[f"Day {TRAIN_DAYS+d+1}" for d in range(PRED_DAYS)]
                     ).to_csv(out / f"{label}.csv")

    # Generate all plots
    print("\nGenerating plots...")
    generate_plots(model, actual, predicted, features, TRAIN_DAYS, PRED_DAYS, out)

    # Generate circuit visualizations
    print("\nGenerating circuit visualizations...")
    visualize_circuit(out)

    # --- Web Data Exports ---
    import json
    web_out = BASE / "qedi_website" / "assets" / "results_data.json"
    tenors_list = []
    maturities_list = []
    for f in features:
        parts = f.split(";")
        t_str = parts[0].replace("Tenor :", "").strip()
        m_str = parts[1].replace("Maturity :", "").strip()
        tenors_list.append(float(t_str))
        maturities_list.append(float(m_str))
        
    export_dict = {
        "features": features,
        "tenors": tenors_list,
        "maturities": maturities_list,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
        "pred_days": PRED_DAYS,
        "train_days": TRAIN_DAYS
    }
    with open(web_out, "w") as f:
        json.dump(export_dict, f)

    anim_out = BASE / "qedi_website" / "assets" / "pca_anim_data.json"
    pc_slice_len = 200
    anim_dict = {
        "true_pc1": model["train_true_pca"][-pc_slice_len:, 0].tolist(),
        "pred_pc1": model["train_pred_pca"][-pc_slice_len:, 0].tolist()
    }
    with open(anim_out, "w") as f:
        json.dump(anim_dict, f)
        
    mae_out = BASE / "qedi_website" / "assets" / "mae_surface.json"
    unique_tenors = sorted(list(set(tenors_list)))
    unique_maturities = sorted(list(set(maturities_list)))
    avg_surface_matrix = np.full((len(unique_tenors), len(unique_maturities)), np.nan)
    avg_errs = np.mean(np.abs(actual - predicted), axis=0)
    for fi, feat in enumerate(features):
        t = tenors_list[fi]
        m = maturities_list[fi]
        avg_surface_matrix[unique_tenors.index(t), unique_maturities.index(m)] = avg_errs[fi]
        
    surface_list = [[None if np.isnan(v) else float(v) for v in row] for row in avg_surface_matrix]
    
    mae_dict = {
        "tenors": unique_tenors,
        "maturities": unique_maturities,
        "surface": surface_list
    }
    with open(mae_out, "w") as f:
        json.dump(mae_dict, f)

    print(f"\nAll saved to {out}/ and JSON outputs written.")
