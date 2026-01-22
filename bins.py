import pandas as pd
from pathlib import Path
import numpy as np

simulation_num = 1  # Change this to process different simulations
input_csv = Path(f"/Users/vignesh/Documents/VoidData/ExcelData/simulation_{simulation_num}_voids.csv")
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

# Create DataFrame with histogram data for normalizing flow model
# Format: one row per bin (0-17) with densitycontrast, radius, and ellipticity columns
histogram_df = pd.DataFrame({
    'densitycontrast': dc_hist,
    'radius': rad_hist,
    'ellipticity': ellip_hist
})

# Save histogram data with index (bin numbers 0-17)
output_dir = Path("/Users/vignesh/Documents/VoidData/BinnedData")
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
output_csv = output_dir / f"simulation_{simulation_num}_binned.csv"
histogram_df.to_csv(output_csv, index=True)

print(f"Binned data saved: {output_csv}")
print(f"\nHistogram shape: {histogram_df.shape}")
print(f"\nFirst few rows:")
print(histogram_df.head())
print(f"\nHistogram statistics:")
print(histogram_df[['densitycontrast', 'radius', 'ellipticity']].describe())

print(f"\nCSV with bins saved: {output_csv}")
print("\nHistogram DataFrame:")
print(histogram_df)