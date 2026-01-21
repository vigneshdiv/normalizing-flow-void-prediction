---
name: Download Globus Files for Simulation 1
overview: Download two specific void data files (shapes_all and centers_all for Quijote simulation 1) from Globus using the CLI, verify paths, and set up the transfer workflow.
todos:
  - id: setup_globus_cli
    content: Verify Globus CLI installation and authentication (globus login)
    status: completed
  - id: discover_dest_endpoint
    content: Find destination endpoint ID (local endpoint or Globus Connect Personal)
    status: completed
  - id: verify_source_paths
    content: List source directory to verify exact file paths and folder structure
    status: completed
  - id: implement_transfer
    content: Create transfer function to download both files using globus transfer command
    status: completed
  - id: monitor_transfers
    content: Add transfer monitoring/wait functionality to ensure downloads complete
    status: completed
  - id: update_extract_paths
    content: Update extract.py to use simulation 1 file paths instead of simulation 0
    status: completed
---

# Download Globus Files for Simulation 1

## Overview

Download `shapes_all_Quijote_1_ss1.0_z0.00_d00.out` and `centers_all_Quijote_1_ss1.0_z0.00_d00.out` from the Gigantes Globus endpoint to your local machine using the Globus CLI.

## Source Information

- **Source Endpoint ID**: `e0eae0aa-5bca-11ea-9683-0e56c063f437` (from the provided Globus link)
- **Collection**: `Quijote_simulations2`
- **Source Directory Path**: `/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/`
- **Full File Paths**:
- `/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/shapes_all_Quijote_1_ss1.0_z0.00_d00.out`
- `/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/centers_all_Quijote_1_ss1.0_z0.00_d00.out`
- **Path Pattern**: For simulation N, the path is `/Gigantes/latin_hypercube/z0.0/sample_Quijote_{N}_ss1.0_z0.00_d00/` where N ranges from 0 to 1999

## Destination Information

- **Destination Directory**: `/Users/vignesh/Documents/VoidData/` (based on existing code in `extract.py`)
- **Destination Endpoint**: To be determined (local endpoint or Globus Connect Personal)

## Implementation Steps

### 1. Setup and Authentication

- Verify Globus CLI is installed (`pip install globus-cli` or `pipx install globus-cli`)
- Authenticate with `globus login`
- Activate source endpoint if needed (`globus endpoint activate`)

### 2. Discover Destination Endpoint

- Check if local endpoint exists: `globus endpoint search "Vignesh"` or `globus endpoint list`
- If using Globus Connect Personal, find the local endpoint ID
- Alternative: Use `globus endpoint search` with your username or check `~/.globus/` config

### 3. Verify Source Path Structure

- List source directory: `globus ls e0eae0aa-5bca-11ea-9683-0e56c063f437:/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/`
- Verify both files exist before transfer:
- `shapes_all_Quijote_1_ss1.0_z0.00_d00.out`
- `centers_all_Quijote_1_ss1.0_z0.00_d00.out`

### 4. Download Files

- Transfer shapes file: `globus transfer SOURCE_EP:/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/shapes_all_Quijote_1_ss1.0_z0.00_d00.out DEST_EP:/Users/vignesh/Documents/VoidData/shapes_all_Quijote_1_ss1.0_z0.00_d00.out`
- Transfer centers file: `globus transfer SOURCE_EP:/Gigantes/latin_hypercube/z0.0/sample_Quijote_1_ss1.0_z0.00_d00/centers_all_Quijote_1_ss1.0_z0.00_d00.out DEST_EP:/Users/vignesh/Documents/VoidData/centers_all_Quijote_1_ss1.0_z0.00_d00.out`
- Monitor transfer status with `globus task wait` or `globus task show`

### 5. Integration with extract.py

- Update `extract.py` to use simulation 1 files instead of simulation 0
- Ensure paths match the downloaded file locations

## Files to Modify

- **[extract.py](extract.py)**: Update file paths from simulation 0 to simulation 1 (lines 4-5)

## Commands Structure

The implementation will include:

1. Helper function to discover/verify destination endpoint
2. Helper function to verify source paths
3. Transfer function using `globus transfer` command
4. Status monitoring for transfer completion
5. Error handling for failed transfers

## Notes

- The plan assumes files will be downloaded to `/Users/vignesh/Documents/VoidData/` to match existing code structure
- If destination endpoint discovery fails, the code will provide clear instructions for manual setup
- Path verification step ensures files exist before attempting transfer
- **Path Pattern**: For simulation N (0-1999), files are located in `/Gigantes/latin_hypercube/z0.0/sample_Quijote_{N}_ss1.0_z0.00_d00/`
- This implementation focuses on simulation 1, but the pattern can be extended for batch downloads of all 2000 simulations