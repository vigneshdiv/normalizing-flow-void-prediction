import pandas as pd
from pathlib import Path
import numpy as np

# cursor may be thinking only to use the dc, rad, and ellip of the 0th sim
# might need to change this fully to print out all sims instead of just 0th for testing

num_bins = 18

# Compute global min/max across all simulations for each property
data_dir = Path("/Users/vignesh/Documents/VoidData/ExcelData")
start_sim = 0
end_sim = 1999
skip_sims = {1897}

# Percentile clipping to remove extreme tails
clip_lo = 0.5
clip_hi = 99.5

all_density = []
all_radius = []
all_ellipticity = []

for sim in range(start_sim, end_sim + 1):
    if sim in skip_sims:
        continue
    sim_file = data_dir / f"simulation_{sim}_voids.csv"
    if not sim_file.exists():
        continue
    df_sim = pd.read_csv(sim_file, usecols=["density_contrast", "radius", "ellipticity"])
    all_density.append(df_sim["density_contrast"].to_numpy())
    all_radius.append(df_sim["radius"].to_numpy())
    all_ellipticity.append(df_sim["ellipticity"].to_numpy())

all_density = np.concatenate(all_density) if all_density else np.array([])
all_radius = np.concatenate(all_radius) if all_radius else np.array([])
all_ellipticity = np.concatenate(all_ellipticity) if all_ellipticity else np.array([])

global_min = {
    "density_contrast": np.percentile(all_density, clip_lo) if all_density.size else None,
    "radius": np.percentile(all_radius, clip_lo) if all_radius.size else None,
    "ellipticity": np.percentile(all_ellipticity, clip_lo) if all_ellipticity.size else None,
}
global_max = {
    "density_contrast": np.percentile(all_density, clip_hi) if all_density.size else None,
    "radius": np.percentile(all_radius, clip_hi) if all_radius.size else None,
    "ellipticity": np.percentile(all_ellipticity, clip_hi) if all_ellipticity.size else None,
}

print("Global min/max values (after percentile clipping):")
for col in global_min.keys():
    print(f"  {col}: min={global_min[col]}, max={global_max[col]}")

# Create bin edges for each property using global min/max
dc_bins = np.linspace(global_min["density_contrast"], global_max["density_contrast"], num_bins + 1)
rad_bins = np.linspace(global_min["radius"], global_max["radius"], num_bins + 1)
ellip_bins = np.linspace(global_min["ellipticity"], global_max["ellipticity"], num_bins + 1)

# Save histogram data with index (bin numbers 0-17)
output_dir = Path("/Users/vignesh/Documents/VoidData/BinnedData")
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

start_sim = 2
end_sim = 1999
processed = 0
skipped = 0

for sim in range(start_sim, end_sim + 1):
    if sim in skip_sims:
        skipped += 1
        continue
    sim_file = data_dir / f"simulation_{sim}_voids.csv"
    if not sim_file.exists():
        skipped += 1
        continue

    df_sim = pd.read_csv(sim_file, usecols=["density_contrast", "radius", "ellipticity"])

    # Use np.histogram with fixed global bin edges
    dc_counts, _ = np.histogram(df_sim["density_contrast"], bins=dc_bins, density=False)
    rad_counts, _ = np.histogram(df_sim["radius"], bins=rad_bins, density=False)
    ellip_counts, _ = np.histogram(df_sim["ellipticity"], bins=ellip_bins, density=False)

    # Normalize to probability distributions
    dc_hist = dc_counts / dc_counts.sum() if dc_counts.sum() > 0 else np.zeros(num_bins)
    rad_hist = rad_counts / rad_counts.sum() if rad_counts.sum() > 0 else np.zeros(num_bins)
    ellip_hist = ellip_counts / ellip_counts.sum() if ellip_counts.sum() > 0 else np.zeros(num_bins)

    # Create DataFrame with histogram data for normalizing flow model
    # Format: one row per bin (0-17) with densitycontrast, radius, and ellipticity columns
    histogram_df = pd.DataFrame({
        "densitycontrast": dc_hist,
        "radius": rad_hist,
        "ellipticity": ellip_hist
    })

    output_csv = output_dir / f"simulation_{sim}_binned.csv"
    histogram_df.to_csv(output_csv, index=True)
    processed += 1

print(f"\nBinned files created: {processed}")
print(f"Skipped simulations: {skipped}")