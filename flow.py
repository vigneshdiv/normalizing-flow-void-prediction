import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import os
import sys
import time

BINNED_DATA_DIR = "/Users/vignesh/Documents/VoidData/BinnedData"
PARAM_FILE = "/Users/vignesh/Documents/VSCodeFiles/NormalizingFlowVoids/latin_hypercube_params.txt"
CHECKPOINT_DIR = "/Users/vignesh/Documents/VSCodeFiles/NormalizingFlowVoids/checkpoints"

NUM_BINS = 18
NUM_SIMS = 2000
PROPERTIES = ["densitycontrast", "radius", "ellipticity"]
PARAM_NAMES = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
NUM_PARAMS = len(PARAM_NAMES)

# "densitycontrast", "radius", "ellipticity", "combined"
MODE = "combined"

STEPS = 1000  # upper bound
LEARNING_RATE = 7e-4
N_CONTEXT_LAYERS = 1
N_CONDITIONAL_LAYERS = 4
SEED = 45
N_POSTERIOR_SAMPLES = 1000
PATIENCE = 50  # stop training if validation loss hasn't improved for this many steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading

def discover_simulations(data_dir, num_sims):
    """Return sorted list of simulation indices that have binned data on disk."""
    available = []
    for i in range(num_sims):
        if os.path.exists(os.path.join(data_dir, f"simulation_{i}_binned.csv")):
            available.append(i)
    return available


def load_binned_data(data_dir, sim_indices, mode):
    """Load histogram vectors for each simulation.

    For single-property modes the vector is length NUM_BINS (18).
    For "combined" mode the vector is densitycontrast|radius|ellipticity (54).
    """
    rows = []
    for idx in sim_indices:
        df = pd.read_csv(
            os.path.join(data_dir, f"simulation_{idx}_binned.csv"), index_col=0
        )
        if mode == "combined":
            vec = np.concatenate([df[p].values for p in PROPERTIES])
        else:
            vec = df[mode].values
        rows.append(vec)
    return np.array(rows)


def load_params(param_file, sim_indices):
    """Load cosmological parameters for the given simulation indices."""
    all_params = np.loadtxt(param_file)
    return all_params[sim_indices]


# model

def create_cond_dist(target_dim, context_dim, n_context_layers, n_cond_layers):
    """Build the conditional spline normalizing flow."""
    base_ctx = dist.Normal(torch.zeros(context_dim), torch.ones(context_dim))
    ctx_transforms = [T.spline_autoregressive(context_dim) for _ in range(n_context_layers)]
    dist_x1 = dist.TransformedDistribution(base_ctx, ctx_transforms)

    base_tgt = dist.Normal(torch.zeros(target_dim), torch.ones(target_dim))
    cond_transforms = [
        T.conditional_spline_autoregressive(target_dim, context_dim=context_dim, bound=5)
        for _ in range(n_cond_layers)
    ]
    dist_x2_given_x1 = dist.ConditionalTransformedDistribution(base_tgt, cond_transforms)

    return dist_x1, dist_x2_given_x1, ctx_transforms, cond_transforms


# data splitting

def prepare_dataset(X, Y, seed=42):
    """70 / 15 / 15 train / valid / test split, deterministic."""
    X_t = torch.tensor(X, dtype=torch.float)
    Y_t = torch.tensor(Y, dtype=torch.float)
    n = X.shape[0]
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)
    n_test = n - n_train - n_valid

    gen = torch.Generator().manual_seed(seed)
    x_splits = torch.utils.data.random_split(X_t, [n_train, n_valid, n_test], generator=gen)
    gen = torch.Generator().manual_seed(seed)
    y_splits = torch.utils.data.random_split(Y_t, [n_train, n_valid, n_test], generator=gen)

    x_train, x_valid, x_test = [torch.stack(list(s)) for s in x_splits]
    y_train, y_valid, y_test = [torch.stack(list(s)) for s in y_splits]
    return x_train, x_valid, x_test, y_train, y_valid, y_test


# training

def train_flow(dist_x1, dist_x2_given_x1, transforms, x_train, y_train,
               steps, lr, x_valid=None, y_valid=None, patience=None):
    modules = torch.nn.ModuleList(transforms)
    optimizer = torch.optim.Adam(modules.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    best_valid_loss = float("inf")
    best_step = 0
    best_state = None
    steps_without_improvement = 0

    for step in range(steps):
        optimizer.zero_grad()
        ln_p_x1 = dist_x1.log_prob(x_train)
        ln_p_x2 = dist_x2_given_x1.condition(x_train.detach()).log_prob(y_train.detach())
        loss = -(ln_p_x1 + ln_p_x2).mean()
        loss.backward()
        optimizer.step()
        dist_x1.clear_cache()
        dist_x2_given_x1.clear_cache()
        train_losses.append(loss.item())

        if x_valid is not None:
            with torch.no_grad():
                vl1 = dist_x1.log_prob(x_valid)
                vl2 = dist_x2_given_x1.condition(x_valid).log_prob(y_valid)
                vloss = -(vl1 + vl2).mean().item()
                valid_losses.append(vloss)
                dist_x1.clear_cache()
                dist_x2_given_x1.clear_cache()

            if vloss < best_valid_loss:
                best_valid_loss = vloss
                best_step = step
                best_state = {k: v.clone() for k, v in modules.state_dict().items()}
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if patience is not None and steps_without_improvement >= patience:
                print(f"  Early stopping at step {step} "
                      f"(no improvement for {patience} steps)")
                break

        if step % 10 == 0:
            msg = f"  step {step:4d} | train loss: {loss.item():.4f}"
            if valid_losses:
                msg += f" | valid loss: {valid_losses[-1]:.4f}"
            print(msg)

    if best_state is not None:
        modules.load_state_dict(best_state)
        print(f"  Restored best model from step {best_step} "
              f"(valid loss: {best_valid_loss:.4f})")

    return train_losses, valid_losses


# evaluation

def evaluate(dist_x2_given_x1, x_test, y_test, y_scaler,
             n_samples=1000, save_dir=None, mode_label=""):
    """Sample posteriors for every test simulation and compare to truth."""
    results = []
    for i in range(x_test.shape[0]):
        samples = dist_x2_given_x1.condition(x_test[i]).sample(torch.Size([n_samples]))
        samples_np = y_scaler.inverse_transform(samples.detach().numpy())
        row = []
        for j in range(NUM_PARAMS):
            row.append(samples_np[:, j].mean())
            row.append(samples_np[:, j].std())
        results.append(row)

    results = np.array(results)
    y_test_np = y_scaler.inverse_transform(y_test.detach().numpy())

    n_show = min(100, x_test.shape[0])
    idxs = np.random.choice(x_test.shape[0], n_show, replace=False)

    metrics_rows = []
    for i, pname in enumerate(PARAM_NAMES):
        true_vals = y_test_np[idxs, i]
        pred_means = results[idxs, 2 * i]
        pred_stds = results[idxs, 2 * i + 1]

        plt.figure(figsize=(6, 5))
        plt.errorbar(true_vals, pred_means, yerr=pred_stds, fmt='o',
                     label='Predicted ± std')
        lims = np.linspace(true_vals.min(), true_vals.max(), 100)
        plt.plot(lims, lims, color='orange', label='Ideal: Pred = True')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{pname} — {mode_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"eval_{mode_label}_{pname}.png"), dpi=150)
        plt.show()

        r2 = r2_score(true_vals, pred_means)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_means))
        chi2 = np.mean(((true_vals - pred_means) ** 2) / (pred_stds ** 2))
        mmre = np.mean(np.abs((true_vals - pred_means) / true_vals))

        metrics_rows.append({
            "parameter": pname,
            "R2": r2,
            "RMSE": rmse,
            "chi_squared": chi2,
            "MMRE": mmre,
        })

        print(f"  {pname}:")
        print(f"    R²:    {r2:.4f}")
        print(f"    RMSE:  {rmse:.4e}")
        print(f"    χ²:    {chi2:.4f}")
        print(f"    MMRE:  {mmre:.4%}")
        print(f"    {'─' * 36}")

    metrics_df = pd.DataFrame(metrics_rows)
    if save_dir:
        metrics_path = os.path.join(save_dir, f"metrics_{mode_label}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Metrics saved → {metrics_path}")

    return results, metrics_df


# checkpointing

def save_checkpoint(transforms, path):
    modules = torch.nn.ModuleList(transforms)
    torch.save(modules.state_dict(), path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(transforms, path):
    modules = torch.nn.ModuleList(transforms)
    modules.load_state_dict(torch.load(path, weights_only=True))
    print(f"  Checkpoint loaded ← {path}")
    return modules


# run one mode

def run(mode):
    print(f"\n{'=' * 60}")
    print(f"  MODE: {mode}")
    print(f"{'=' * 60}")

    input_dim = NUM_BINS * len(PROPERTIES) if mode == "combined" else NUM_BINS

    sim_indices = discover_simulations(BINNED_DATA_DIR, NUM_SIMS)
    print(f"  Simulations found: {len(sim_indices)}")

    X = load_binned_data(BINNED_DATA_DIR, sim_indices, mode)
    Y = load_params(PARAM_FILE, sim_indices)
    print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")

    x_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    y_scaler = StandardScaler()
    Y = y_scaler.fit_transform(Y)

    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_dataset(X, Y, SEED)
    print(f"  Train: {x_train.shape[0]}  Valid: {x_valid.shape[0]}  Test: {x_test.shape[0]}")

    dist_x1, dist_x2_given_x1, ctx_tf, cond_tf = create_cond_dist(
        NUM_PARAMS, input_dim, N_CONTEXT_LAYERS, N_CONDITIONAL_LAYERS
    )
    all_transforms = ctx_tf + cond_tf

    # train
    t0 = time.time()
    train_losses, valid_losses = train_flow(
        dist_x1, dist_x2_given_x1, all_transforms,
        x_train, y_train, STEPS, LEARNING_RATE, x_valid, y_valid,
        patience=PATIENCE
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    if valid_losses:
        plt.plot(valid_losses, label="Validation")
        best_idx = int(np.argmin(valid_losses))
        plt.axvline(best_idx, color="gray", linestyle="--", alpha=0.7,
                     label=f"Best valid (step {best_idx})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve — {mode}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    mode_dir = os.path.join(CHECKPOINT_DIR, mode)
    os.makedirs(mode_dir, exist_ok=True)
    plt.savefig(os.path.join(mode_dir, f"loss_{mode}.png"), dpi=150)
    plt.show()

    # checkpoint
    save_checkpoint(all_transforms, os.path.join(mode_dir, f"flow_{mode}.pt"))

    # evaluate
    print(f"\n  Evaluation on test set ({x_test.shape[0]} simulations):")
    _, metrics_df = evaluate(
        dist_x2_given_x1, x_test, y_test, y_scaler,
        n_samples=N_POSTERIOR_SAMPLES, save_dir=mode_dir, mode_label=mode
    )

    return dist_x1, dist_x2_given_x1, all_transforms, y_scaler


# entry point

if __name__ == "__main__":
    VALID_MODES = PROPERTIES + ["combined"]
    modes = sys.argv[1:] if len(sys.argv) > 1 else [MODE]
    for m in modes:
        if m not in VALID_MODES:
            sys.exit(f"Invalid mode '{m}'. Choose from: {VALID_MODES}")
        run(m)