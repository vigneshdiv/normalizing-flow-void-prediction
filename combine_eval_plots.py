"""
Build a combined evaluation figure for the well-constrained parameters
Omega_m and sigma_8.

Layout: two panels side by side. Parameter names are drawn inside
each panel (no (a)/(b) subcaptions).

Rather than stitching the rasterised PNGs written by flow.py, this regenerates
the panels from the saved checkpoint and draws them as vectors.

Input:  checkpoints/<mode>/flow_<mode>.pt
Output: checkpoints/<mode>/eval_all_<mode>.pdf (and .png)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import flow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Two square panels side by side.
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.4
FONT_PT = 10

PANELS = [
    ("Omega_m", r"$\Omega_m$"),
    ("sigma_8", r"$\sigma_8$"),
]


def posterior_moments(mode):
    """Load the saved flow and return (true, mean, std) arrays for the test set."""
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

    means, stds = [], []
    for i in range(x_test.shape[0]):
        samples = dist_x2_given_x1.condition(x_test[i]).sample(
            torch.Size([flow.N_POSTERIOR_SAMPLES])
        )
        samples = y_scaler.inverse_transform(samples.detach().numpy())
        means.append(samples.mean(axis=0))
        stds.append(samples.std(axis=0))

    true = y_scaler.inverse_transform(y_test.detach().numpy())
    return true, np.array(means), np.array(stds)


def main(mode="combined"):
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "legend.fontsize": FONT_PT - 1,
    })

    true, means, stds = posterior_moments(mode)

    # Same 100-simulation subsample flow.py plots; numpy's stream is seeded in
    # run() and untouched by torch, so this reproduces it exactly.
    np.random.seed(flow.SEED)
    n_show = min(100, true.shape[0])
    idxs = np.random.choice(true.shape[0], n_show, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14, wspace=0.28)

    for ax, (pname, label) in zip(axes, PANELS):
        j = flow.PARAM_NAMES.index(pname)
        t = true[idxs, j]
        m = means[idxs, j]
        s = stds[idxs, j]

        ax.errorbar(t, m, yerr=s, fmt="o", markersize=2.5, elinewidth=0.7,
                    capsize=0, label=r"Predicted $\pm$ std")
        lims = np.linspace(t.min(), t.max(), 100)
        ax.plot(lims, lims, color="orange", linewidth=1.2,
                label="Ideal: Pred = True")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_box_aspect(1)
        ax.legend(loc="upper left", frameon=True, handlelength=1.4,
                  borderpad=0.4, labelspacing=0.3)
        ax.text(0.97, 0.04, label, transform=ax.transAxes,
                fontsize=FONT_PT + 6, va="bottom", ha="right")

    # Saved without bbox_inches="tight" so the margins above are preserved and
    # the output width is exactly FIG_WIDTH_IN.
    out_pdf = os.path.join(BASE_DIR, "checkpoints", mode, f"eval_all_{mode}.pdf")
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_pdf.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "combined")
