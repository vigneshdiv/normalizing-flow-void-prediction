import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

input_csv = Path("/Users/vignesh/Documents/VoidData/simulation_0_voids.csv")
df = pd.read_csv(input_csv)

num_bins = 18

# Create bin edges for each property using numpy linspace
dc_bins = np.linspace(df['density_contrast'].min(), df['density_contrast'].max(), num_bins + 1)
rad_bins = np.linspace(df['radius'].min(), df['radius'].max(), num_bins + 1)
ellip_bins = np.linspace(df['ellipticity'].min(), df['ellipticity'].max(), num_bins + 1)

# Use pd.cut to bin the data and get normalized frequencies (probabilities that sum to 1)
dc_cut = pd.cut(df['density_contrast'], bins=dc_bins, include_lowest=True, labels=False, duplicates='drop')
dc_counts = dc_cut.value_counts(normalize=True).sort_index()
dc_hist = np.zeros(num_bins)
for idx, val in dc_counts.items():
    if 0 <= idx < num_bins:
        dc_hist[int(idx)] = val

rad_cut = pd.cut(df['radius'], bins=rad_bins, include_lowest=True, labels=False, duplicates='drop')
rad_counts = rad_cut.value_counts(normalize=True).sort_index()
rad_hist = np.zeros(num_bins)
for idx, val in rad_counts.items():
    if 0 <= idx < num_bins:
        rad_hist[int(idx)] = val

ellip_cut = pd.cut(df['ellipticity'], bins=ellip_bins, include_lowest=True, labels=False, duplicates='drop')
ellip_counts = ellip_cut.value_counts(normalize=True).sort_index()
ellip_hist = np.zeros(num_bins)
for idx, val in ellip_counts.items():
    if 0 <= idx < num_bins:
        ellip_hist[int(idx)] = val

# Create bin centers for plotting and data representation
dc_edges = dc_bins
rad_edges = rad_bins
ellip_edges = ellip_bins
dc_centers = (dc_edges[:-1] + dc_edges[1:]) / 2
rad_centers = (rad_edges[:-1] + rad_edges[1:]) / 2
ellip_centers = (ellip_edges[:-1] + ellip_edges[1:]) / 2

# Create DataFrame with histogram data for normalizing flow model
# Format: one row per bin (0-17) with densitycontrast, radius, and ellipticity columns
histogram_df = pd.DataFrame({
    'densitycontrast': dc_hist,
    'radius': rad_hist,
    'ellipticity': ellip_hist
})

# Save histogram data with index (bin numbers 0-17)
output_csv = Path("/Users/vignesh/Documents/VoidData/simulation_0_histograms.csv")
histogram_df.to_csv(output_csv, index=True)

print(f"Histogram data saved: {output_csv}")
print(f"\nHistogram shape: {histogram_df.shape}")
print(f"\nFirst few rows:")
print(histogram_df.head())
print(f"\nHistogram statistics:")
print(histogram_df[['densitycontrast', 'radius', 'ellipticity']].describe())

# Create visualization plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Density Contrast histogram
axes[0].bar(dc_centers, dc_hist, width=np.diff(dc_edges), alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Density Contrast')
axes[0].set_ylabel('Normalized Density')
axes[0].set_title('Density Contrast Distribution')
axes[0].grid(True, alpha=0.3)

# Radius histogram
axes[1].bar(rad_centers, rad_hist, width=np.diff(rad_edges), alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('Radius')
axes[1].set_ylabel('Normalized Density')
axes[1].set_title('Radius Distribution')
axes[1].grid(True, alpha=0.3)

# Ellipticity histogram
axes[2].bar(ellip_centers, ellip_hist, width=np.diff(ellip_edges), alpha=0.7, edgecolor='black', color='green')
axes[2].set_xlabel('Ellipticity')
axes[2].set_ylabel('Normalized Density')
axes[2].set_title('Ellipticity Distribution')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path("/Users/vignesh/Documents/VoidData/simulation_0_histograms.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nHistogram plots saved: {plot_path}")
plt.close()

print(f"\nCSV with bins saved: {output_csv}")
print("\nHistogram DataFrame:")
print(histogram_df)