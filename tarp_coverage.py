"""
TARP (Tests of Accuracy with Random Points) coverage testing for the trained
normalizing flow model.

Usage: python tarp_coverage.py [mode]
Output: checkpoints/<mode>/tarp_coverage.pdf

Requires: pip install tarp
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tarp import get_tarp_coverage

import flow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Well-constrained parameters only; Omega_b, h, n_s are dropped from the figure.
TARP_PARAMS = ["Omega_m", "sigma_8"]
PARAM_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "sigma_8": r"$\sigma_8$",
}


def load_posterior_samples(mode="combined", n_samples=1000):
    """Load checkpoint and generate posterior samples for the full test set.
    
    Returns:
        samples: shape (n_samples, n_test, n_params)
        theta_true: shape (n_test, n_params)
    """
    sim_indices = flow.discover_simulations(flow.BINNED_DATA_DIR, flow.NUM_SIMS)
    X = flow.load_binned_data(flow.BINNED_DATA_DIR, sim_indices, mode)
    Y = flow.load_params(flow.PARAM_FILE, sim_indices)
    X = StandardScaler().fit_transform(X)
    y_scaler = StandardScaler()
    Y = y_scaler.fit_transform(Y)

    _, _, x_test, _, _, y_test = flow.prepare_dataset(X, Y, flow.SEED)

    ckpt = os.path.join(BASE_DIR, "checkpoints", mode, f"flow_{mode}.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"Missing checkpoint: {ckpt}\nRun flow.py {mode} first.")

    torch.manual_seed(flow.SEED)
    dist_x2_given_x1, cond_tf = flow.create_cond_dist(
        flow.NUM_PARAMS, X.shape[1], flow.N_CONDITIONAL_LAYERS
    )
    modules = torch.nn.ModuleList(cond_tf)
    modules.load_state_dict(torch.load(ckpt, weights_only=True))

    n_test = x_test.shape[0]
    print(f"Generating {n_samples} posterior samples for {n_test} test simulations...")
    
    # Shape: (n_test, n_samples, n_params)
    all_samples = []
    for i in range(n_test):
        samples = dist_x2_given_x1.condition(x_test[i]).sample(torch.Size([n_samples]))
        samples = y_scaler.inverse_transform(samples.detach().numpy())
        all_samples.append(samples)
    
    # Stack and transpose to (n_samples, n_test, n_params) as tarp expects
    all_samples = np.stack(all_samples, axis=0)  # (n_test, n_samples, n_params)
    all_samples = np.transpose(all_samples, (1, 0, 2))  # (n_samples, n_test, n_params)
    
    theta_true = y_scaler.inverse_transform(y_test.detach().numpy())  # (n_test, n_params)
    
    return all_samples, theta_true


def run_tarp_per_parameter(samples, theta_true):
    """Run TARP separately for each parameter.
    
    Returns dict mapping parameter name to (ecp, alpha) tuple.
    """
    results = {}
    for pname in TARP_PARAMS:
        i = flow.PARAM_NAMES.index(pname)
        print(f"  Running TARP for {pname}...")
        # Extract single parameter: (n_samples, n_test, 1)
        param_samples = samples[:, :, i:i+1]
        param_theta = theta_true[:, i:i+1]
        
        ecp, alpha = get_tarp_coverage(
            param_samples, param_theta,
            references="random",
            metric="euclidean",
            bootstrap=False,
            seed=flow.SEED,
        )
        results[pname] = (ecp, alpha)
    return results


def _on_curve(alpha, ecp, x):
    return (float(x), float(np.interp(x, alpha, ecp)))


def _most_separated(alpha, ecp_self, ecp_other, lo, hi):
    """x in [lo, hi] where |this curve - the other| is largest."""
    grid = np.linspace(lo, hi, 80)
    self_y = np.interp(grid, alpha, ecp_self)
    other_y = np.interp(grid, alpha, ecp_other)
    k = int(np.argmax(np.abs(self_y - other_y)))
    return float(grid[k]), float(self_y[k])


def plot_tarp(results, save_path):
    """Plot expected vs empirical coverage for Omega_m and sigma_8."""
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 11,
    })
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    
    colors = {"Omega_m": "#1f77b4", "sigma_8": "#9467bd"}
    om_a, om_e = results["Omega_m"]
    s8_a, s8_e = results["sigma_8"]
    ax.plot(om_a, om_e, color=colors["Omega_m"], linewidth=1.5,
            label=PARAM_LABELS["Omega_m"])
    ax.plot(s8_a, s8_e, color=colors["sigma_8"], linewidth=1.5,
            label=PARAM_LABELS["sigma_8"])

    s8_on_om = np.interp(om_a, s8_a, s8_e)
    # Window chosen where the two curves usually peel apart, not where they
    # cross the diagonal together.
    om_xy = _most_separated(om_a, om_e, s8_on_om, 0.30, 0.40)
    s8_xy = _on_curve(s8_a, s8_e, 0.4)

    ax.annotate(
        PARAM_LABELS["Omega_m"],
        xy=om_xy, xytext=(om_xy[0] - 0.08, om_xy[1] + 0.12),
        color=colors["Omega_m"], fontsize=13, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=colors["Omega_m"],
                        lw=1.1, shrinkA=3, shrinkB=1),
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.9),
    )
    ax.annotate(
        PARAM_LABELS["sigma_8"],
        xy=s8_xy, xytext=(0.58, 0.18),
        color=colors["sigma_8"], fontsize=13, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=colors["sigma_8"],
                        lw=1.1, shrinkA=3, shrinkB=1),
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.9),
    )
    
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("TARP Coverage Test")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.replace('.pdf', '.png')}")


def main(mode="combined"):
    samples, theta_true = load_posterior_samples(mode)
    print(f"Samples shape: {samples.shape}")
    print(f"Theta true shape: {theta_true.shape}")
    
    results = run_tarp_per_parameter(samples, theta_true)
    
    # Print summary statistics
    print("\nTARP Summary:")
    print(f"{'Parameter':<12} {'ATC (area to curve)':<20}")
    print("-" * 35)
    for pname in TARP_PARAMS:
        ecp, alpha = results[pname]
        # ATC: area between ECP curve and diagonal for alpha > 0.5
        mask = alpha > 0.5
        atc = np.trapezoid(ecp[mask] - alpha[mask], alpha[mask])
        print(f"{pname:<12} {atc:+.4f}")
    
    save_path = os.path.join(BASE_DIR, "checkpoints", mode, "tarp_coverage.pdf")
    plot_tarp(results, save_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "combined")
