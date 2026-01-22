from pathlib import Path
import pandas as pd

# Directory paths
SHAPES_DIR = Path("/Users/vignesh/Documents/VoidData/ShapesAll")
CENTERS_DIR = Path("/Users/vignesh/Documents/VoidData/CentersAll")
OUTPUT_DIR = Path("/Users/vignesh/Documents/VoidData/ExcelData")

# Process simulations 2 through 1999 (skipping 0 and 1 since they already exist)
START_SIM = 1898
END_SIM = 1999

print(f"Processing simulations {START_SIM} to {END_SIM}...")
print(f"Shapes directory: {SHAPES_DIR}")
print(f"Centers directory: {CENTERS_DIR}")
print(f"Output directory: {OUTPUT_DIR}\n")

successful = 0

failed = 0

for simulation in range(START_SIM, END_SIM + 1):
    try:
        # Construct file paths
        shapes_file = SHAPES_DIR / f"shapes_all_Quijote_{simulation}_ss1.0_z0.00_d00.out"
        centers_file = CENTERS_DIR / f"centers_all_Quijote_{simulation}_ss1.0_z0.00_d00.out"
        output_csv = OUTPUT_DIR / f"simulation_{simulation}_voids.csv"
        
        # Check if output already exists
        if output_csv.exists():
            print(f"[{simulation}/{END_SIM}] Simulation {simulation}: Output already exists, skipping...")
            successful += 1
            continue
        
        # Check if input files exist
        if not shapes_file.exists():
            print(f"[{simulation}/{END_SIM}] Simulation {simulation}: Shapes file not found, skipping...")
            failed += 1
            continue
        
        if not centers_file.exists():
            print(f"[{simulation}/{END_SIM}] Simulation {simulation}: Centers file not found, skipping...")
            failed += 1
            continue
        
        print(f"[{simulation}/{END_SIM}] Processing simulation {simulation}...")
        
        # Read shapes file
        shapes_data = []
        with open(shapes_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    void_id = int(parts[0])
                    ellipticity = float(parts[1])
                    shapes_data.append([void_id, ellipticity])
        
        shapes_df = pd.DataFrame(shapes_data, columns=["void_id", "ellipticity"])
        
        # Read centers file
        centers_data = []
        with open(centers_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 9:
                    radius = float(parts[4])
                    void_id = int(parts[7])
                    density_contrast = float(parts[8])
                    centers_data.append([void_id, density_contrast, radius])
        
        centers_df = pd.DataFrame(centers_data, columns=["void_id", "density_contrast", "radius"])
        
        # Merge the two dataframes on void_id using inner join
        merged_df = pd.merge(centers_df, shapes_df, on="void_id", how="inner")
        
        # Save the merged dataframe to CSV
        merged_df.to_csv(output_csv, index=False)
        
        print(f"  ✓ Saved: {output_csv} ({len(merged_df)} voids)")
        successful += 1
        
        # Print progress every 100 simulations
        if simulation % 100 == 0:
            print(f"\nProgress: {simulation - START_SIM + 1} simulations processed\n")
            
    except Exception as e:
        print(f"[{simulation}/{END_SIM}] Simulation {simulation}: Error - {e}")
        failed += 1
        continue

print("\n" + "=" * 60)
print("Extraction Summary")
print("=" * 60)
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Total processed: {successful + failed}")
print(f"Output files saved to: {OUTPUT_DIR}")
print("=" * 60)
