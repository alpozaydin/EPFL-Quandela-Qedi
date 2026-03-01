#!/usr/bin/env python3
"""
Temporal QRC: Memory Modes + Virtual Nodes for Swaption Pricing
================================================================
Inspired by Li et al. "QRC for Realized Volatility Forecasting" (2024),
adapted from qubit-based Ising Hamiltonian to photonic circuits.

Key improvements over final_model.py:
  1. DEDICATED MEMORY MODES: 8 modes = 5 input + 3 memory
     - Only input modes (0-4) get phase-shift encoding
     - Memory modes (5-7) accumulate temporal info across layers
     (original model: 6 modes, 5 encoded, 1 memory)

  2. VIRTUAL NODES: Multiple post-processing depths (1, 2, 3 interferometers)
     - Each depth samples the quantum state at a different evolution stage
     - Mimics the paper's δτ sub-evolution measurements
     - Total circuits = N_RESERVOIRS × N_VIRTUAL_NODES

  3. FEATURE SELECTION: Mutual information ranking of reservoir features
     - Identifies which quantum features are actually informative
     - Optional pruning to reduce feature dimensionality

  4. EXPLICIT TEMPORAL STRUCTURE: Layer i = time step i
     - Day t-4's PCA → mixing interferometer → Day t-3's PCA → ... → measure
     - Memory modes carry forward quantum correlations across time steps

Pipeline:
  Raw prices → StandardScaler → PCA(5) → sliding window(5) →
  [3 seeds × 3 virtual depths = 9 circuits] Temporal QRC + LexGrouping →
  + raw PCA features → Ridge → PCA inverse → Scaler inverse → prices
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
from sklearn.feature_selection import mutual_info_regression

import perceval as pcvl
from merlin import QuantumLayer, ComputationSpace, LexGrouping

# ── Config ───────────────────────────────────────────────────────────────
N_PCA           = 5
WINDOW          = 5
N_INPUT_MODES   = 5      # modes that receive input encoding (= N_PCA)
N_MEMORY_MODES  = 3      # dedicated memory modes (never encoded)
N_MODES         = N_INPUT_MODES + N_MEMORY_MODES  # 8
N_PHOTONS       = 3
N_RESERVOIRS    = 3      # different random seeds
N_VIRTUAL_NODES = 3      # post-processing depths (1, 2, 3)
RIDGE_ALPHA     = 0.1
USE_LEXGROUPING = True   # compress reservoir output
LEX_OUT         = 10     # LexGrouping output dimension per circuit

TOTAL_ENC   = N_INPUT_MODES * WINDOW  # 25
INPUT_STATE = [1] * N_PHOTONS + [0] * (N_MODES - N_PHOTONS)

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "CHALLENGE RESOURCES" / "DATASETS"
OUT  = BASE / "quantum_results" / "temporal_qrc"
OUT.mkdir(parents=True, exist_ok=True)


# =====================================================================
# BUILDING BLOCKS
# =====================================================================

def build_temporal_circuit(n_modes, n_input_modes, n_steps, n_virtual_depth):
    """
    Build a temporal photonic circuit with dedicated memory modes.

    Architecture:
      For each time step t = 0..n_steps-1:
        [Trainable interferometer on ALL n_modes]
        [Input phase shifters on modes 0..n_input_modes-1 ONLY]
      For each virtual node layer v = 0..n_virtual_depth-1:
        [Trainable interferometer on ALL n_modes]

    The memory modes (n_input_modes .. n_modes-1) are NEVER directly
    encoded — they accumulate temporal information purely through the
    mixing interferometers, analogous to the paper's "memory qubits"
    that preserve quantum state via Hamiltonian evolution.

    The virtual node layers add post-processing depth, analogous to
    the paper's δτ = τ/VirtualNode sub-evolution measurements.
    """
    circuit = pcvl.Circuit(n_modes)
    c = 0

    # ── Temporal encoding layers (one per time step) ──────────────────
    for t in range(n_steps):
        # Mixing interferometer across ALL modes (input + memory)
        # This entangles input information with memory modes
        circuit.add(0, pcvl.GenericInterferometer(
            n_modes,
            lambda i, _t=t: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_i{i}"))
                              // pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_o{i}"))),
            shape=pcvl.InterferometerShape.RECTANGLE,
        ))
        # Input encoding: phase shifters on INPUT MODES ONLY
        # Memory modes are left untouched — they carry forward info
        for m in range(n_input_modes):
            circuit.add(m, pcvl.PS(pcvl.P(f"input{c}")))
            c += 1

    # ── Virtual node post-processing layers ───────────────────────────
    # Each adds a mixing interferometer WITHOUT any input encoding
    # Different depths give different "snapshots" of quantum evolution
    for v in range(n_virtual_depth):
        circuit.add(0, pcvl.GenericInterferometer(
            n_modes,
            lambda i, _v=v: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_i{i}"))
                              // pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_o{i}"))),
            shape=pcvl.InterferometerShape.RECTANGLE,
        ))

    return circuit, c  # c = n_input_modes * n_steps = 25


def build_reservoirs():
    """
    Build ensemble: N_RESERVOIRS seeds × N_VIRTUAL_NODES depths.

    For each seed, all virtual node depths share the SAME temporal
    encoding parameters (same reservoir), but differ in post-processing
    depth (different virtual node measurements).
    """
    reservoirs = []
    for r_seed in range(N_RESERVOIRS):
        for vd in range(1, N_VIRTUAL_NODES + 1):
            torch.manual_seed(42 + r_seed * 1000)
            circ, n_enc = build_temporal_circuit(
                N_MODES, N_INPUT_MODES, WINDOW, vd
            )
            core = QuantumLayer(
                input_size=n_enc,
                circuit=circ,
                input_state=INPUT_STATE,
                input_parameters=["input"],
                trainable_parameters=["t"],
                computation_space=ComputationSpace.UNBUNCHED,
                dtype=torch.float32,
            )
            if USE_LEXGROUPING:
                layer = nn.Sequential(core, LexGrouping(core.output_size, LEX_OUT))
            else:
                layer = core
            layer.eval()
            reservoirs.append(layer)
    return reservoirs


def quantum_features(x, reservoirs):
    """Pass a 25-dim window through all reservoir circuits → concatenated features."""
    if len(x) < TOTAL_ENC:
        x = np.pad(x, (0, TOTAL_ENC - len(x)))
    elif len(x) > TOTAL_ENC:
        x = x[:TOTAL_ENC]
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        feats = []
        for r in reservoirs:
            out = r(xt)
            if out.is_complex():
                out = out.real
            feats.append(out.squeeze(0).numpy())
        return np.concatenate(feats)


def make_windows(pc_data):
    """Create (X, y) pairs from PCA time series."""
    X, y = [], []
    for t in range(WINDOW, len(pc_data)):
        X.append(pc_data[t - WINDOW: t].flatten())
        y.append(pc_data[t])
    return np.array(X), np.array(y)


def feature_importance_analysis(Q, y, feature_names=None):
    """Compute mutual information between each quantum feature and targets."""
    mi_scores = np.zeros(Q.shape[1])
    for c in range(y.shape[1]):
        mi = mutual_info_regression(Q, y[:, c], random_state=42)
        mi_scores += mi
    mi_scores /= y.shape[1]
    return mi_scores


# =====================================================================
# TRAIN
# =====================================================================

def train(prices_train):
    """
    Train the temporal QRC pipeline.

    Returns:
        model dict with all fitted components + training diagnostics
    """
    # 1. Fit scaler & PCA
    scaler = StandardScaler().fit(prices_train)
    scaled = scaler.transform(prices_train)
    pca = PCA(n_components=N_PCA).fit(scaled)
    pc = pca.transform(scaled)

    # 2. Build temporal reservoirs
    print("  Building temporal reservoirs...")
    reservoirs = build_reservoirs()
    n_circuits = len(reservoirs)
    print(f"  Built {n_circuits} circuits "
          f"({N_RESERVOIRS} seeds × {N_VIRTUAL_NODES} virtual depths)")

    # 3. Create windows
    X, y = make_windows(pc)

    # 4. Extract quantum features
    print(f"  Extracting features for {len(X)} windows...")
    Q = np.array([quantum_features(x, reservoirs) for x in X])
    print(f"  Quantum features shape: {Q.shape}")

    # 5. Feature importance (for diagnostics, not pruning)
    mi_scores = feature_importance_analysis(Q, y)

    # 6. Concatenate quantum + raw PCA features
    features = np.hstack([Q, X])
    print(f"  Total feature dim: {features.shape[1]} "
          f"(quantum={Q.shape[1]} + raw={X.shape[1]})")

    # 7. Fit Ridge
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(features, y)

    # 8. Training diagnostics
    y_pred_pca = ridge.predict(features)
    y_pred_prices = scaler.inverse_transform(pca.inverse_transform(y_pred_pca))
    y_true_prices = scaler.inverse_transform(pca.inverse_transform(y))

    train_mse = mean_squared_error(y_true_prices, y_pred_prices)
    train_r2 = r2_score(y_true_prices.flatten(), y_pred_prices.flatten())
    print(f"  Training MSE: {train_mse:.2e}, R²: {train_r2:.6f}")

    return {
        "scaler": scaler,
        "pca": pca,
        "reservoirs": reservoirs,
        "ridge": ridge,
        "last_pc": pc[-WINDOW:],
        # diagnostics
        "train_true_pca": y,
        "train_pred_pca": y_pred_pca,
        "train_true_prices": y_true_prices,
        "train_pred_prices": y_pred_prices,
        "pca_explained_var": pca.explained_variance_ratio_,
        "mi_scores": mi_scores,
    }


# =====================================================================
# PREDICT
# =====================================================================

def predict(model, n_days):
    """Autoregressive prediction for n_days into the future."""
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
# PLOTS
# =====================================================================

def generate_plots(model, actual, predicted, naive, feat_cols,
                   train_days, pred_days, out):
    """Generate all diagnostic and comparison plots."""

    # ── 1. Scatter: Predicted vs Actual ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual.flatten(), predicted.flatten(), s=3, alpha=0.4, color="#3498db")
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    r2 = r2_score(actual.flatten(), predicted.flatten())
    mse = mean_squared_error(actual.flatten(), predicted.flatten())
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Temporal QRC: R²={r2:.6f}, MSE={mse:.2e}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=150)
    plt.close(fig)
    print("  [1/6] Scatter plot saved")

    # ── 2. Per-day MSE comparison: Model vs Naive ─────────────────────
    days_labels = [f"Day {train_days+d+1}" for d in range(pred_days)]
    mse_model = [mean_squared_error(actual[d], predicted[d]) for d in range(pred_days)]
    mse_naive = [mean_squared_error(actual[d], naive[d]) for d in range(pred_days)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(pred_days)
    ax.bar(x - 0.2, mse_model, 0.35, label="Temporal QRC", color="#3498db")
    ax.bar(x + 0.2, mse_naive, 0.35, label="Naive (repeat last day)", color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(days_labels)
    ax.set_ylabel("MSE")
    ax.set_title("Per-Day MSE: Temporal QRC vs Naive Baseline")
    ax.legend()
    for i in range(pred_days):
        if mse_model[i] < mse_naive[i]:
            ax.text(i - 0.2, mse_model[i], "✓", ha="center", va="bottom",
                    fontsize=12, color="green", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison_vs_naive.png", dpi=150)
    plt.close(fig)
    print("  [2/6] Model vs Naive comparison saved")

    # ── 3. Error Heatmap ──────────────────────────────────────────────
    errors = predicted - actual
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(errors, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(errors)), vmax=np.max(np.abs(errors)))
    ax.set_yticks(range(pred_days))
    ax.set_yticklabels(days_labels)
    ax.set_xlabel("Feature index (224 swaption columns)")
    ax.set_title("Prediction Error Heatmap (Predicted − Actual)")
    plt.colorbar(im, ax=ax, label="Error", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "error_heatmap.png", dpi=150)
    plt.close(fig)
    print("  [3/6] Error heatmap saved")

    # ── 4. Training Fit in PCA Space ──────────────────────────────────
    y_true_pca = model["train_true_pca"]
    y_pred_pca = model["train_pred_pca"]

    fig, axes = plt.subplots(N_PCA, 1, figsize=(14, 3 * N_PCA), sharex=True)
    for c in range(N_PCA):
        axes[c].plot(y_true_pca[:, c], color="#2c3e50", linewidth=0.8,
                     label="Actual", alpha=0.8)
        axes[c].plot(y_pred_pca[:, c], color="#e74c3c", linewidth=0.8,
                     label="Predicted", alpha=0.7)
        r2c = r2_score(y_true_pca[:, c], y_pred_pca[:, c])
        axes[c].set_ylabel(f"PC{c+1}")
        axes[c].set_title(f"PC{c+1}  (train R² = {r2c:.6f})", fontsize=10)
        axes[c].legend(fontsize=8)
    axes[-1].set_xlabel("Training sample index")
    fig.suptitle("Training Fit in PCA Space (Temporal QRC)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "training_fit_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] Training fit (PCA) saved")

    # ── 5. Feature Importance (Mutual Information) ────────────────────
    mi_scores = model["mi_scores"]
    n_q = len(mi_scores)
    q_per_circuit = n_q // (N_RESERVOIRS * N_VIRTUAL_NODES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # All features
    axes[0].bar(range(n_q), mi_scores, color="#3498db", alpha=0.7, width=1.0)
    axes[0].set_xlabel("Quantum Feature Index")
    axes[0].set_ylabel("Mutual Information (avg across targets)")
    axes[0].set_title(f"Feature Importance — All {n_q} Quantum Features")

    # Grouped by circuit
    circuit_mi = []
    labels = []
    for r in range(N_RESERVOIRS):
        for v in range(N_VIRTUAL_NODES):
            idx = (r * N_VIRTUAL_NODES + v) * q_per_circuit
            circuit_mi.append(np.mean(mi_scores[idx:idx + q_per_circuit]))
            labels.append(f"S{r+1}\nV{v+1}")

    colors_v = plt.cm.Blues(np.linspace(0.3, 0.9, N_VIRTUAL_NODES))
    bar_colors = [colors_v[v % N_VIRTUAL_NODES] for _ in range(N_RESERVOIRS)
                  for v in range(N_VIRTUAL_NODES)]
    axes[1].bar(range(len(circuit_mi)), circuit_mi, color=bar_colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Circuit (Seed × Virtual Depth)")
    axes[1].set_ylabel("Mean MI")
    axes[1].set_title("Feature Importance by Circuit\n(S=seed, V=virtual depth)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  [5/6] Feature importance plot saved")

    # ── 6. Training Residual Distribution ─────────────────────────────
    y_true_px = model["train_true_prices"]
    y_pred_px = model["train_pred_prices"]
    residuals = (y_true_px - y_pred_px).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(residuals, bins=80, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (actual − predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Residuals (mean={np.mean(residuals):.2e}, "
                      f"std={np.std(residuals):.2e})")
    axes[1].hist(np.abs(residuals), bins=80, color="#e67e22", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("|Residual|")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"|Residuals| (median={np.median(np.abs(residuals)):.2e})")
    fig.tight_layout()
    fig.savefig(out / "training_residuals.png", dpi=150)
    plt.close(fig)
    print("  [6/6] Training residuals saved")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TEMPORAL QRC: Memory Modes + Virtual Nodes")
    print("=" * 70)
    print(f"  Config:")
    print(f"    Modes:       {N_MODES} ({N_INPUT_MODES} input + {N_MEMORY_MODES} memory)")
    print(f"    Photons:     {N_PHOTONS}")
    print(f"    Virtual:     {N_VIRTUAL_NODES} depths")
    print(f"    Reservoirs:  {N_RESERVOIRS} seeds")
    print(f"    Total circ:  {N_RESERVOIRS * N_VIRTUAL_NODES}")
    print(f"    LexGrouping: {USE_LEXGROUPING} (out={LEX_OUT})")
    print(f"    Ridge α:     {RIDGE_ALPHA}")
    print()

    # Load data
    df = pd.read_excel(DATA / "train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c != "Date"]
    prices = df[feat_cols].values.astype(np.float64)

    TRAIN_DAYS = 450
    PRED_DAYS = 6

    # ── Train ─────────────────────────────────────────────────────────
    print(f"Training on days 1-{TRAIN_DAYS}...")
    model = train(prices[:TRAIN_DAYS])
    print()

    # ── Predict ───────────────────────────────────────────────────────
    print(f"Predicting days {TRAIN_DAYS+1}-{TRAIN_DAYS+PRED_DAYS}...")
    predicted = predict(model, PRED_DAYS)
    actual = prices[TRAIN_DAYS:TRAIN_DAYS+PRED_DAYS]

    # ── Naive baseline (repeat last known day) ────────────────────────
    naive = np.tile(prices[TRAIN_DAYS - 1], (PRED_DAYS, 1))
    naive_mse = mean_squared_error(actual.flatten(), naive.flatten())
    naive_r2 = r2_score(actual.flatten(), naive.flatten())

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'Day':<6} {'Model MSE':>14} {'Naive MSE':>14} "
          f"{'MAE':>10} {'RelErr%':>10}")
    print("─" * 60)
    wins = 0
    for d in range(PRED_DAYS):
        mse_m = mean_squared_error(actual[d], predicted[d])
        mse_n = mean_squared_error(actual[d], naive[d])
        mae_m = mean_absolute_error(actual[d], predicted[d])
        rel = 100 * np.mean(np.abs(actual[d] - predicted[d])
                            / (np.abs(actual[d]) + 1e-10))
        marker = " ✓" if mse_m < mse_n else ""
        if mse_m < mse_n:
            wins += 1
        print(f"{TRAIN_DAYS+d+1:<6} {mse_m:>14.2e} {mse_n:>14.2e} "
              f"{mae_m:>10.6f} {rel:>9.4f}%{marker}")

    overall_mse = mean_squared_error(actual.flatten(), predicted.flatten())
    overall_r2 = r2_score(actual.flatten(), predicted.flatten())

    print(f"\n  Model:  MSE={overall_mse:.2e}  R²={overall_r2:.6f}")
    print(f"  Naive:  MSE={naive_mse:.2e}  R²={naive_r2:.6f}")
    print(f"  Wins:   {wins}/{PRED_DAYS} days")

    if overall_mse < naive_mse:
        print("  ✅ Temporal QRC BEATS naive baseline!")
    else:
        gap = (overall_mse / naive_mse - 1) * 100
        print(f"  ❌ Temporal QRC below naive (gap: {gap:.2f}%)")

    # ── Save CSVs ─────────────────────────────────────────────────────
    for label, data in [("actual", actual), ("predicted", predicted)]:
        pd.DataFrame(data, columns=feat_cols,
                     index=[f"Day {TRAIN_DAYS+d+1}" for d in range(PRED_DAYS)]
                     ).to_csv(OUT / f"{label}.csv")

    import json
    anim_out = BASE / "qedi_website" / "assets" / "price_anim_data.json"
    pc_slice_len = 200
    anim_dict = {
        "true_price": model["train_true_prices"][-pc_slice_len:, 0].tolist(),
        "pred_price": model["train_pred_prices"][-pc_slice_len:, 0].tolist()
    }
    with open(anim_out, "w") as f:
        json.dump(anim_dict, f)

    # ── Generate Plots ────────────────────────────────────────────────
    print("\nGenerating plots...")
    generate_plots(model, actual, predicted, naive, feat_cols,
                   TRAIN_DAYS, PRED_DAYS, OUT)

    print(f"\nAll saved to {OUT}/")
    print("=" * 70)
