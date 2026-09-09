"""
Joint Omega_m–sigma_8 posterior contours for four well-spread test cosmologies.

Layout: 2x2, one panel per cosmology. Each panel shows KDE contours of the
conditional flow posterior with the true (Omega_m, sigma_8) marked.

Test-set indices (SEED=45 split, 301 test simulations):
  253  low  Omega_m, low  sigma_8   (0.1521, 0.6559)
   53  low  Omega_m, high sigma_8   (0.1393, 0.9531)
  131  high Omega_m, low  sigma_8   (0.4673, 0.6417)
   91  high Omega_m, high sigma_8   (0.4583, 0.9653)

Chosen as the nearest unused test points to the 10th/90th-percentile corners
of the test-set Omega_m–sigma_8 plane.

Usage: python joint_posterior.py [mode]
Output: checkpoints/<mode>/joint_posterior_om_s8.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

import flow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_SAMPLES = 8000
OM_IDX = flow.PARAM_NAMES.index("Omega_m")
S8_IDX = flow.PARAM_NAMES.index("sigma_8")

# Test-set indices into the SEED=45 70/15/15 split (not simulation IDs).
SELECTED = [219, 53, 131, 91]

FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 6.4
FONT_PT = 10


def load_flow_and_test(mode):
    """Reuse flow.py's data split, scaler, and checkpoint loading."""
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

    y_test_np = y_scaler.inverse_transform(y_test.detach().numpy())
    return dist_x2_given_x1, y_scaler, x_test, y_test_np


def sample_posterior(dist_x2_given_x1, y_scaler, x_cond, n_samples):
    """Same sampling path as flow.evaluate: condition, sample, inverse-scale."""
    samples = dist_x2_given_x1.condition(x_cond).sample(torch.Size([n_samples]))
    return y_scaler.inverse_transform(samples.detach().numpy())


def main(mode="combined"):
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
    })

    dist_x2, y_scaler, x_test, y_test_np = load_flow_and_test(mode)
    n_test = x_test.shape[0]
    for i in SELECTED:
        if not 0 <= i < n_test:
            sys.exit(f"Selected index {i} is outside the test set of size {n_test}.")

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
                             sharex=False, sharey=False)
    axes = axes.ravel()

    torch.manual_seed(flow.SEED)
    for ax, idx in zip(axes, SELECTED):
        samples = sample_posterior(dist_x2, y_scaler, x_test[idx], N_SAMPLES)
        om = samples[:, OM_IDX]
        s8 = samples[:, S8_IDX]
        om_true = y_test_np[idx, OM_IDX]
        s8_true = y_test_np[idx, S8_IDX]

        sns.kdeplot(
            x=om, y=s8, ax=ax, fill=True, thresh=0.05, levels=6,
            cmap="Blues", alpha=0.85,
        )
        sns.kdeplot(
            x=om, y=s8, ax=ax, fill=False, thresh=0.05, levels=6,
            color="#1f4e79", linewidths=0.8,
        )
        ax.plot(om_true, s8_true, "x", color="crimson", markersize=9,
                markeredgewidth=1.8, zorder=5)
        # Numeric truth label goes in the empty corner of this panel,
        # opposite the contour, so it is readable against white space.
        if om_true < 0.3:
            ha, x_frac = ("right", 0.97)
        else:
            ha, x_frac = ("left", 0.03)
        if s8_true < 0.8:
            va, y_frac = ("top", 0.97)
        else:
            va, y_frac = ("bottom", 0.03)
        ax.text(
            x_frac, y_frac,
            rf"$\Omega_m$ = {om_true:.2f}, $\sigma_8$ = {s8_true:.2f}",
            transform=ax.transAxes, ha=ha, va=va,
            fontsize=FONT_PT + 1, color="crimson",
        )
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_ylabel(r"$\sigma_8$")
        ax.set_xlim(0.0, 0.6)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        ax.set_ylim(top=1.2)

        print(f"  test idx {idx:3d}  true  Omega_m={om_true:.4f}  "
              f"sigma_8={s8_true:.4f}  |  posterior mean "
              f"Omega_m={om.mean():.4f}  sigma_8={s8.mean():.4f}")

    fig.tight_layout()
    out = os.path.join(BASE_DIR, "checkpoints", mode, "joint_posterior_om_s8.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "combined")
